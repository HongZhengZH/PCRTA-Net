import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer Attention Layer
class TransformerAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1).to(x.device)  # (seq_len, batch_size, C)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x.permute(1, 2, 0).view(b, c, d, h, w)


# Transformer-based Channel Attention Block (CAB)
class TransformerCAB(nn.Module):
    def __init__(self, n_feat, num_heads=4, dropout=0.1):
        super(TransformerCAB, self).__init__()
        self.conv1 = nn.Conv3d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.transformer = TransformerAttentionLayer(n_feat, num_heads, dropout)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        res = self.transformer(res)
        return res + x


# Modified LeNet with Transformer Attention
class TransformerLeNet(nn.Module):
    def __init__(self):
        super(TransformerLeNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_feat = 32
        self.kernel_size = 3
        self.num_heads = 4
        self.head = nn.Conv3d(13, self.n_feat, self.kernel_size, padding=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm3d(self.n_feat).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1).to(self.device)
        self.CAB1 = TransformerCAB(self.n_feat, self.num_heads).to(self.device)
        self.CAB2 = TransformerCAB(self.n_feat, self.num_heads).to(self.device)
        self.CAB3 = TransformerCAB(self.n_feat, self.num_heads).to(self.device)
        self.CAB4 = TransformerCAB(self.n_feat, self.num_heads).to(self.device)
        self.tail = nn.Conv3d(self.n_feat, 2, kernel_size=1, stride=2, bias=False).to(self.device)

    def forward(self, x, clinical_datas):
        x = x.to(self.device)
        clinical_datas = clinical_datas.to(self.device)
        x_size = x.shape
        clinical_datas = clinical_datas.view(clinical_datas.shape[0], clinical_datas.shape[2])

        # Concatenate clinical data across channels
        for i in range(clinical_datas.shape[1]):
            sigma_all = clinical_datas[:, i].view(-1, 1, 1, 1, 1).repeat(1, 1, x_size[2], x_size[3], x_size[4])
            x = torch.cat((x, sigma_all), 1)

        x = self.head(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.CAB1(x)
        x = self.maxpool(x)
        x = self.CAB2(x)
        x = self.maxpool(x)
        x = self.CAB3(x)
        x = self.maxpool(x)
        x = self.CAB4(x)
        x = self.tail(x)

        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x)
        return x





