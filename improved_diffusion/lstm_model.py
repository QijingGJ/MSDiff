
import torch.nn as nn
import torchvision.models as modelss
from einops import rearrange


class Resnet_LSTM(nn.Module):
    def __init__(self, hidden_dim=768, n_layers=2, n_class=256):
        super().__init__()
        # ------------------------------
        # 1. ResNet特征提取（共享参数）
        # ------------------------------
        resnet = modelss.resnet18(pretrained=True)

        # 移除全连接层，保留卷积特征提取能力
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 输出 [batch*seq, 512, 4, 4]
        self.resnet[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # ------------------------------
        # 2. LSTM时序建模（共享参数）
        # ------------------------------
        self.lstm = nn.LSTM(
            input_size=512 * 4 * 4,  # ResNet最终特征图展平后的维度
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        # ------------------------------
        # 3. 多分支输出层
        # ------------------------------
        self.main_head = nn.Linear(hidden_dim, n_class)
        self.minus_head = nn.Linear(hidden_dim, n_class)
        self.plus_head = nn.Linear(hidden_dim, n_class)

    def forward(self, x, if_aux='main'):
        """输入格式: (batch_size, seq_len, 1, 128, 128)"""
        # ------------------------------
        # 特征提取
        # ------------------------------
        batch, seq_len = x.size(0), x.size(1)

        # 合并批次和序列维度
        x = rearrange(x, "b s c h w -> (b s) c h w")  # [batch*seq, 1, 128, 128]

        # 通过ResNet主干
        features = self.resnet(x)  # [batch*seq, 512, 4, 4]

        # 展平特征
        features = features.view(features.size(0), -1)  # [batch*seq, 512*4*4]

        # ------------------------------
        # 时序建模
        # ------------------------------
        features = features.view(batch, seq_len, -1)  # [batch, seq_len, 8192]
        lstm_out, _ = self.lstm(features)  # [batch, seq_len, hidden_dim]

        # ------------------------------
        # 多分支输出
        # ------------------------------
        if if_aux == 'main':
            out = self.main_head(lstm_out)
        elif if_aux == 'minus':
            out = self.minus_head(lstm_out)
        elif if_aux == 'plus':
            out = self.plus_head(lstm_out)
        else:
            raise ValueError("Invalid if_aux mode")

        return out  # [batch, seq_len, 256]


