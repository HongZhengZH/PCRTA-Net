import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transform

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, aug_mode, clinical):
        self.images_path = images_path
        self.images_class = images_class
        self.aug_mode = aug_mode
        self.clinical = clinical

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        str_list = self.images_path[item]
        str_split = str_list.split('/')
        image_str_t1 = self.images_path[item] + '/'+ str_split[3] + '.nii.gz'
        mask_str_t1 = self.images_path[item] + '/'+ str_split[3] + '-label.nii.gz'
        def get_brain_region(image,mask):
            indice_list = np.where(mask > 0)
            channel_0_min = min(indice_list[0])
            channel_0_max = max(indice_list[0])

            channel_1_min = min(indice_list[1])
            channel_1_max = max(indice_list[1])

            channel_2_min = min(indice_list[2])
            channel_2_max = max(indice_list[2])

            brain_volume = image[channel_0_min:channel_0_max, channel_1_min:channel_1_max,channel_2_min:channel_2_max]
            return brain_volume

        def img_mask(image_str,mask_str,trunc_min,trunc_max):
            img = sitk.ReadImage(image_str,sitk.sitkFloat32)
            img = sitk.GetArrayFromImage(img)
            img =  torch.tensor(img,dtype=torch.float32)
            img[img <= trunc_min] = trunc_min  # 截断
            img[img >= trunc_max] = trunc_max
            img = (img - trunc_min) / (trunc_max - trunc_min)
            mask = sitk.ReadImage(mask_str,sitk.sitkFloat32)
            mask = sitk.GetArrayFromImage(mask)
            mask = torch.tensor(mask, dtype=torch.float32)
            img = img*mask

            # 将脑部抠出
            out_img = get_brain_region(img,mask)
            c, h, w = out_img.shape
            CROP_SIZE = 32  # 128
            CROP_SIZE_z = 32  #  32
            if w < CROP_SIZE or h < CROP_SIZE or c < CROP_SIZE_z:
                # zero cropping
                pad_c = (CROP_SIZE_z - c) if (c < CROP_SIZE_z) else 0
                pad_h = (CROP_SIZE - h) if (h < CROP_SIZE) else 0
                pad_w = (CROP_SIZE - w) if (w < CROP_SIZE) else 0
                rem_c = pad_c % 2
                rem_h = pad_h % 2
                rem_w = pad_w % 2
                pad_dim_c = (pad_c // 2, pad_c // 2 + rem_c)
                pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
                pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
                npad = (pad_dim_c, pad_dim_h, pad_dim_w)
                img_pad = np.pad(out_img, npad, 'constant', constant_values=0)
                c, h, w = img_pad.shape
            else:
                img_pad = out_img
            # center crop
            c_offset = (c - CROP_SIZE_z) // 2
            h_offset = (h - CROP_SIZE) // 2
            w_offset = (w - CROP_SIZE) // 2

            cropped_img = img_pad[c_offset:(c_offset + CROP_SIZE_z) ,h_offset:(h_offset + CROP_SIZE),
                          w_offset:(w_offset + CROP_SIZE)]
            img = torch.unsqueeze(torch.from_numpy(np.array(cropped_img)), dim=0)

            return img

        # 拼装四种img
        img = img_mask(image_str_t1, mask_str_t1,-1150,350)   # A1=[0,300]
        img = img.cpu().data.numpy()

        # data augment
        if self.aug_mode==1:
            if random.random() < 0.5:  # random_horizontal_flip
                img = img[:, :, :, ::-1].copy()

            if random.random() < 0.5:  # random_vertical_flip
                img = img[:, :, ::-1, :].copy()

            if random.random() < 0.5:  # random_vertical_flip
                img = img[:, ::-1, :, :].copy()

        img = torch.from_numpy(img)
        label = self.images_class[item]

        patient_id = str_split[3]
        X = self.clinical.values[0:, 2:]
        y = self.clinical.values[0:, 0:1]
        position = np.where(y == patient_id)
        clinical_data = X[position[0],:]
        clinical_data = torch.from_numpy(clinical_data.astype(float))   # shape = 18 X 1
        clinical_data = clinical_data.to(torch.float32)

        return img, label, clinical_data

    @staticmethod
    def collate_fn(batch):
        images, labels, clinical_datas = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        clinical_datas = torch.stack(clinical_datas, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, clinical_datas

