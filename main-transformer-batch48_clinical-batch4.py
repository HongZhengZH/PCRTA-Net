import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from torchvision import transforms
import tqdm
from data_3_clinical import MyDataSet
from data_loader_seed import split_data
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import  roc_curve, auc
import math
import pandas as pd
from sklearn.metrics import classification_report,roc_curve,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from model_transformer_clinical import TransformerLeNet
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch. cuda.is_available() else "cpu")
print("using {} device.".format(device))
####################################################################################
# 超参数定义
model_str = './model_str/'  # 所有K-fold训练好的模型存放的位置
data_list_txt_str = './data_list_txt_str/'  # 所有K-fold数据List和打乱的真题数据存放的位置
traindata_str = 'train_data.txt'  # 所有训练过程存放的位置
testdata_str = 'test_data.txt'  # 所有训练过程存放的位置
net_name = 'Transformer-batch4-clinical'      # 保存的网络的名称
num_workers = 0   # 多线程的数量，默认O
batch_size = 4    # 4
epochs = 50      #    默认 100
learning_rate = 0.0001  # 学习率
num_class = 2           # 分类的类别数目
run_mode = 'test'       # 运行模式，train=训练，test=测试
test_train_or_test = 'test' # 判断是测试训练集还是验证集
df = pd.read_csv("./filtered_sub_solid_Lungnodule.csv")
####################################################################################
# 创建文件夹
if not os.path.exists(model_str):
        os.makedirs(model_str)
####################################################################################
def result_print(train_Label, y_pred, valid_preds_fold):
    xg_eval_auc = metrics.roc_auc_score(train_Label, valid_preds_fold, average='weighted')  # 验证集上的auc值
    xg_eval_acc = metrics.accuracy_score(train_Label, y_pred)  # 验证集的准确度
    mcm = confusion_matrix(train_Label, y_pred)
    # print(mcm)
    TN = mcm[0, 0]
    FP = mcm[0, 1]
    FN = mcm[1, 0]
    TP = mcm[1, 1]
    xg_eval_specificity = TN/(TN+FP)
    xg_eval_sensitivity = TP / (TP + FN)
    xg_eval_precision = TP/(TP+FP)
    print('|eval_auc=%.4f' % np.array(xg_eval_auc),'|eval_accuracy=%.4f' % np.array(xg_eval_acc),
          '|eval_sensitivity=%.4f' % np.array(xg_eval_sensitivity),
          '|eval_specificity=%.4f' % np.array(xg_eval_specificity),
          '|eval_precision=%.4f' % np.array(xg_eval_precision))
    # print(classification_report(train_Label, y_pred, digits=4))
    return xg_eval_auc,xg_eval_acc,xg_eval_sensitivity,xg_eval_specificity,xg_eval_precision
##############################################################################
# 加载数据
train_images_path, train_images_label, val_images_path, val_images_label = split_data(traindata_str,testdata_str,data_list_txt_str)
print(np.array(val_images_path).shape)

data_transform = {
        "train": transforms.ToTensor(),
        "val": transforms.ToTensor()
}
train_data_set = MyDataSet(images_path=train_images_path,
                           images_class=train_images_label, aug_mode=1, clinical = df)

train_loader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=train_data_set.collate_fn)

val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label, aug_mode=0,clinical = df)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
                                         num_workers=num_workers,collate_fn = val_dataset.collate_fn)

net = TransformerLeNet()

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))

net.to(device)
loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([2, 1])).float().to(device))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
val_num = len(val_dataset)

best_auc = 0.000
best_epoch = 0
train_steps = len(train_loader)
train_inf = []
if not os.path.exists(model_str+ net_name):
        os.makedirs(model_str+ net_name)

save_path = model_str + net_name +  '/model.pth'
if run_mode == 'train':
    for epoch in range(epochs):
        print('epoch= ', epoch)
        net.train()
        running_loss = 0.0

        for step, data in enumerate(train_loader):
            images, labels, clinical_datas = data

            outputs = net(images.to(device), clinical_datas.to(device))
            loss = loss_function(outputs, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        train_loader_desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                    epochs,
                                                                    running_loss)
        print(train_loader_desc)

        # validate
        net.eval()

        test_loss = 0

        valid_preds_fold = np.zeros(len(val_images_label))
        target_all = []
        with torch.no_grad():
            for i, (inputs, targets, clinical_datas) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs, clinical_datas.to(device))
                target_all.append(targets.cpu().detach().numpy())

                loss = loss_function(outputs, targets)
                test_loss += loss.item()

                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = outputs.cpu().detach().numpy()[:, 1]

        print('Test loss=', test_loss)

        threshold = 0.5
        result_test = []
        for pred in valid_preds_fold:
            result_test.append(1 if pred > threshold else 0)

        xg_eval_auc,xg_eval_acc,xg_eval_sensitivity,xg_eval_specificity,xg_eval_precision = result_print(
            val_images_label, result_test,valid_preds_fold)

        xg_eval_best = xg_eval_auc

        # saved best model
        if xg_eval_best > best_auc:
            print('saved model', epoch)
            best_epoch = epoch
            best_auc = xg_eval_best
            torch.save(net.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            save_path_epoch = model_str + net_name + '/' + str(epoch + 1) + '_seed.pth'
            torch.save(net.state_dict(), save_path_epoch)

        # save result
            output_data = [epoch + 1, best_auc, xg_eval_auc]
            output_file_name = model_str + net_name+ "/train.txt"
            output_file = open(output_file_name, 'a')
            for fp in output_data:  # write data in txt
                output_file.write(str(fp))
                output_file.write(',')
            output_file.write('\n')  # line feed
            output_file.close()

        print( 'best epoch = ', best_epoch,' best_auc=', best_auc)
####################################################################################################
if run_mode == 'test':
    if test_train_or_test == 'train': # 判断是测试训练集还是验证集
        train_data_set = MyDataSet(images_path=train_images_path,
                                   images_class=train_images_label, aug_mode=0, clinical=df)

        train_loader = torch.utils.data.DataLoader(train_data_set,
                                                   batch_size=batch_size,
                                                   # shuffle=True,
                                                   num_workers=num_workers,
                                                   collate_fn=train_data_set.collate_fn)
        val_images_label = train_images_label
        val_loader = train_loader
    net.load_state_dict(torch.load(save_path))
    net.eval()
    test_loss = 0
    valid_preds_fold = np.zeros(len(val_images_label))
    target_all = []
    with torch.no_grad():
        for i, (inputs, targets, clinical_datas) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, clinical_datas.to(device))
            target_all.append(targets.cpu().detach().numpy())

            loss = loss_function(outputs, targets)
            test_loss += loss.item()

            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = outputs.cpu().detach().numpy()[:, 1]

    print('Test loss=', test_loss)

    threshold = 0.1  #  默认0.5
    result_test = []
    for pred in valid_preds_fold:
        result_test.append(1 if pred > threshold else 0)

    xg_eval_auc, xg_eval_acc, xg_eval_sensitivity, xg_eval_specificity, xg_eval_precision = result_print(
        val_images_label, result_test, valid_preds_fold)

    xg_eval_best = (xg_eval_auc + xg_eval_acc + xg_eval_sensitivity + xg_eval_specificity) / 4

    # save result
    output_data = [val_images_label, valid_preds_fold]
    if test_train_or_test == 'train':  # 判断是测试训练集还是验证集
        output_file_name = model_str + net_name + "/predict_train.csv"
    else:
        output_file_name = model_str + net_name + "/predict_test.csv"
    np.savetxt(output_file_name, output_data, delimiter=',', fmt='%s')
    #######################################################################################
    # 绘制ROC曲线图
    label_size = 13
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 11.5,
             }
    plt.figure(figsize=(5.5, 5))
    fpr, tpr, thresholds = roc_curve(val_images_label, valid_preds_fold)  # ravel()表示平铺开来
    plt.plot(fpr, tpr, label='Radiomics (AUC=%.4f)' % xg_eval_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('1 - Specificity', font1)
    plt.ylabel('Sensitivity', font1)
    axes = plt.gca()
    axes.set_xlim([-0.05, 1.05])
    axes.set_ylim([-0.05, 1.05])
    plt.legend(loc=4, prop=font2)
    plt.tick_params(labelsize=label_size)
    labels = axes.get_xticklabels() + axes.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()

print('Finished Training')