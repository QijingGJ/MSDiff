import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRoutingGate(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

        # 时间步嵌入修正：使用Linear层处理时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(128, 128),  # 输入应为嵌入向量维度
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        # 特征压缩模块
        self.feature_compressor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU()
        )

        # 动态权重预测器
        self.initial_weight_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim + 128, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels-1, 1),
        )

        # 条件特征投影
        self.condition_projectors = nn.ModuleList([
            nn.Conv2d(1, in_channels-1, 3, padding=1) for _ in range(in_channels-1)
        ])

    def forward(self, conditions, noisy_img, t):
        # 时间嵌入处理修正
        B = t.shape[0]
        t_emb = self.time_embed(t)  # [B, 128]
        t_emb = t_emb.view(B, -1, 1, 1)  # [B, 128, 1, 1]

        # 特征压缩
        x = torch.cat([*conditions, noisy_img], dim=1)  # (16, 5, 128, 128)
        spatial_feat = self.feature_compressor(x)  # [B, 64, 128, 128]

        # 时间信息融合修正
        t_emb_expanded = t_emb.expand(-1, -1, 128, 128)  # [B, 128, 128, 128]
        fused_feat = torch.cat([spatial_feat, t_emb_expanded], dim=1)  # (B, 192, 128, 128)

        # 初始权重预测
        logits = self.initial_weight_predictor(fused_feat)  # (16, 4, 128, 128)
        weight_maps = torch.softmax(logits, dim=1)  # (16, 4, 128, 128)

        # 投影条件特征  list: each (16, 4, 128, 128)
        proj_conditions = [proj(cond) for proj, cond in zip(self.condition_projectors, conditions)]

        # 动态路由迭代修正
        for _ in range(self.num_iterations - 1):
            # 加权融合  (16, 4, 128, 128)
            weighted_sum = sum([w.unsqueeze(1) * c for w, c in zip(weight_maps.unbind(1), proj_conditions)])

            similarity_maps = []
            for cond in proj_conditions:
                cond_norm = F.normalize(cond, dim=1)
                weighted_sum_norm = F.normalize(weighted_sum, dim=1)
                similarity = (cond_norm * weighted_sum_norm).sum(dim=1, keepdim=True)
                similarity_maps.append(similarity)

            # 累积更新logits
            logits = logits + torch.cat(similarity_maps, dim=1)
            weight_maps = torch.softmax(logits, dim=1)

        # 最终融合
        fused_conditions = sum([w.unsqueeze(1) * c for w, c in zip(weight_maps.unbind(1), proj_conditions)])

        return torch.cat([fused_conditions, noisy_img], dim=1), weight_maps



