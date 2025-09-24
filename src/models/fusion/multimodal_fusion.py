import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from config.model_config import ModelConfig


class CrossAttentionLayer(nn.Module):
    """交叉注意力层 - 让两个模态互相关注"""

    def __init__(self, query_dim: int, key_value_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_value_dim, hidden_dim)
        self.value_proj = nn.Linear(key_value_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_len, query_dim]
            key_value: [batch_size, kv_len, key_value_dim]
            mask: [batch_size, query_len, kv_len] (optional)

        Returns:
            output: [batch_size, query_len, query_dim]
        """
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]

        # Linear projections
        Q = self.query_proj(query)  # [batch_size, query_len, hidden_dim]
        K = self.key_proj(key_value)  # [batch_size, kv_len, hidden_dim]
        V = self.value_proj(key_value)  # [batch_size, kv_len, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, query_len, head_dim]
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)     # [batch_size, num_heads, kv_len, head_dim]
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)     # [batch_size, num_heads, kv_len, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, query_len, kv_len]

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, query_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)  # [batch_size, query_len, hidden_dim]

        # Output projection
        output = self.output_proj(context)

        # Residual connection and layer norm
        output = self.layer_norm(query + output)

        return output


class BiModalFusionBlock(nn.Module):
    """双模态融合块"""

    def __init__(self, image_dim: int, brep_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # 图像关注Brep
        self.image_cross_attention = CrossAttentionLayer(
            query_dim=image_dim,
            key_value_dim=brep_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Brep关注图像
        self.brep_cross_attention = CrossAttentionLayer(
            query_dim=brep_dim,
            key_value_dim=image_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 自注意力层
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.brep_self_attention = nn.MultiheadAttention(
            embed_dim=brep_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward网络
        self.image_ffn = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_dim)
        )

        self.brep_ffn = nn.Sequential(
            nn.Linear(brep_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, brep_dim)
        )

        self.image_norm = nn.LayerNorm(image_dim)
        self.brep_norm = nn.LayerNorm(brep_dim)

    def forward(self, image_features: torch.Tensor, brep_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: [batch_size, image_seq_len, image_dim]
            brep_features: [batch_size, brep_seq_len, brep_dim]

        Returns:
            Tuple of (enhanced_image_features, enhanced_brep_features)
        """
        # 交叉注意力
        image_cross = self.image_cross_attention(image_features, brep_features)
        brep_cross = self.brep_cross_attention(brep_features, image_features)

        # 自注意力
        image_self, _ = self.image_self_attention(image_cross, image_cross, image_cross)
        brep_self, _ = self.brep_self_attention(brep_cross, brep_cross, brep_cross)

        # Feed-forward
        image_out = self.image_norm(image_self + self.image_ffn(image_self))
        brep_out = self.brep_norm(brep_self + self.brep_ffn(brep_self))

        return image_out, brep_out


class BiModalFuser(nn.Module):
    """双模态融合器 - 融合图像和Brep特征"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 输入投影层 - 将特征投影到相同的隐藏维度
        self.image_input_proj = nn.Linear(config.image_dim, config.fusion_dim)
        self.brep_input_proj = nn.Linear(config.brep_dim, config.fusion_dim)

        # 多层融合块
        self.fusion_blocks = nn.ModuleList([
            BiModalFusionBlock(
                image_dim=config.fusion_dim,
                brep_dim=config.fusion_dim,
                hidden_dim=config.fusion_dim * 2,
                num_heads=config.num_attention_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_transformer_layers)
        ])

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout)
        )

        # 模态特定的输出投影
        self.image_output_proj = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.brep_output_proj = nn.Linear(config.fusion_dim, config.fusion_dim)

    def forward(self, image_features: torch.Tensor, brep_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_features: [batch_size, image_dim] 或 [batch_size, seq_len, image_dim]
            brep_features: [batch_size, brep_dim] 或 [batch_size, seq_len, brep_dim]

        Returns:
            Dict包含:
                - fused_features: 融合后的特征 [batch_size, fusion_dim]
                - image_enhanced: 增强的图像特征
                - brep_enhanced: 增强的Brep特征
        """
        # 验证输入维度
        if len(image_features.shape) not in [2, 3]:
            raise ValueError(f"Image features must be 2D or 3D, got shape: {image_features.shape}")
        if len(brep_features.shape) not in [2, 3]:
            raise ValueError(f"Brep features must be 2D or 3D, got shape: {brep_features.shape}")

        # 记录原始是否为2D输入（用于简化处理）
        image_was_2d = len(image_features.shape) == 2
        brep_was_2d = len(brep_features.shape) == 2

        # 确保输入是3D张量 [batch_size, seq_len, feature_dim]
        if image_was_2d:
            image_features = image_features.unsqueeze(1)  # [batch_size, 1, image_dim]
        if brep_was_2d:
            brep_features = brep_features.unsqueeze(1)    # [batch_size, 1, brep_dim]

        # 输入投影
        image_projected = self.image_input_proj(image_features)  # [batch_size, seq_len, fusion_dim]
        brep_projected = self.brep_input_proj(brep_features)     # [batch_size, seq_len, fusion_dim]

        # 如果都是2D输入且序列长度为1，使用简化路径
        if image_was_2d and brep_was_2d and image_projected.shape[1] == 1 and brep_projected.shape[1] == 1:
            # 直接处理2D特征，避免不必要的序列处理
            image_enhanced = image_projected.squeeze(1)  # [batch_size, fusion_dim]
            brep_enhanced = brep_projected.squeeze(1)    # [batch_size, fusion_dim]
        else:
            # 通过融合块
            image_enhanced = image_projected
            brep_enhanced = brep_projected

            for fusion_block in self.fusion_blocks:
                image_enhanced, brep_enhanced = fusion_block(image_enhanced, brep_enhanced)

            # 全局池化（如果有序列维度）
            if image_enhanced.shape[1] > 1:
                image_enhanced = image_enhanced.mean(dim=1)  # [batch_size, fusion_dim]
            else:
                image_enhanced = image_enhanced.squeeze(1)

            if brep_enhanced.shape[1] > 1:
                brep_enhanced = brep_enhanced.mean(dim=1)    # [batch_size, fusion_dim]
            else:
                brep_enhanced = brep_enhanced.squeeze(1)

        # 最终融合
        concatenated = torch.cat([image_enhanced, brep_enhanced], dim=-1)  # [batch_size, fusion_dim * 2]
        fused_features = self.final_fusion(concatenated)  # [batch_size, fusion_dim]

        # 模态特定的输出
        image_output = self.image_output_proj(image_enhanced)
        brep_output = self.brep_output_proj(brep_enhanced)

        return {
            'fused_features': fused_features,
            'image_enhanced': image_output,
            'brep_enhanced': brep_output,
            'attention_weights': None  # 可以在需要时添加注意力权重
        }


def create_bimodal_fuser(config: Optional[ModelConfig] = None) -> BiModalFuser:
    """创建双模态融合器的工厂函数"""
    return BiModalFuser(config)


if __name__ == "__main__":
    # 测试代码
    config = ModelConfig()
    fuser = create_bimodal_fuser(config)

    # 模拟输入
    batch_size = 4
    image_features = torch.randn(batch_size, config.image_dim)   # [4, 768]
    brep_features = torch.randn(batch_size, config.brep_dim)     # [4, 1024]

    # 前向传播测试
    with torch.no_grad():
        output = fuser(image_features, brep_features)
        print(f"Fused features shape: {output['fused_features'].shape}")
        print(f"Enhanced image features shape: {output['image_enhanced'].shape}")
        print(f"Enhanced brep features shape: {output['brep_enhanced'].shape}")
        print("Bimodal fusion test passed!")