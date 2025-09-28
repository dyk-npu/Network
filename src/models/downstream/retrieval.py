import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from config.model_config import ModelConfig


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.norm1(query + self.dropout(attn_output))

        # Feed forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + self.dropout(ffn_output))

        return output


class AdvancedSimilarityComputation(nn.Module):
    """高级相似度计算模块"""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        # Advanced similarity networks
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Multi-scale similarity computation
        self.local_attention = MultiheadAttention(d_model, num_heads, batch_first=True)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Similarity fusion
        self.similarity_fusion = nn.Sequential(
            nn.Linear(3, 64),  # 3 types of similarities
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        计算高级相似度

        Args:
            query_features: [batch_size, d_model]
            candidate_features: [num_candidates, d_model]

        Returns:
            similarities: [batch_size, num_candidates]
        """
        batch_size = query_features.size(0)
        num_candidates = candidate_features.size(0)

        # Project features
        query_proj = self.query_projection(query_features)  # [batch_size, d_model]
        key_proj = self.key_projection(candidate_features)  # [num_candidates, d_model]

        # 1. Cosine similarity with learnable temperature
        query_norm = F.normalize(query_proj, p=2, dim=-1)
        key_norm = F.normalize(key_proj, p=2, dim=-1)
        cosine_sim = torch.mm(query_norm, key_norm.t()) * torch.exp(self.temperature)

        # 2. Euclidean distance based similarity
        query_expanded = query_proj.unsqueeze(1).expand(-1, num_candidates, -1)
        key_expanded = key_proj.unsqueeze(0).expand(batch_size, -1, -1)
        euclidean_dist = torch.norm(query_expanded - key_expanded, p=2, dim=-1)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)

        # 3. Learned similarity (attention-based)
        query_for_attn = query_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        key_for_attn = key_proj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_candidates, d_model]

        attn_output, attn_weights = self.local_attention(
            query_for_attn, key_for_attn, key_for_attn
        )
        learned_sim = attn_weights.squeeze(1)  # [batch_size, num_candidates]

        # Combine similarities
        similarities_stack = torch.stack([cosine_sim, euclidean_sim, learned_sim], dim=-1)
        final_similarities = self.similarity_fusion(similarities_stack).squeeze(-1)

        return final_similarities


class HardNegativeMiner(nn.Module):
    """难样本挖掘模块"""

    def __init__(self, margin: float = 0.2, hard_ratio: float = 0.3):
        super().__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio

    def mine_hard_negatives(self, query_features: torch.Tensor,
                           candidate_features: torch.Tensor,
                           positive_indices: torch.Tensor,
                           num_negatives: int = 10) -> torch.Tensor:
        """
        挖掘难样本负例

        Args:
            query_features: [batch_size, d_model]
            candidate_features: [num_candidates, d_model]
            positive_indices: [batch_size] 正样本索引
            num_negatives: 每个查询选择的负样本数量

        Returns:
            negative_indices: [batch_size, num_negatives]
        """
        batch_size = query_features.size(0)
        num_candidates = candidate_features.size(0)

        # 计算所有相似度
        similarities = torch.mm(
            F.normalize(query_features, p=2, dim=-1),
            F.normalize(candidate_features, p=2, dim=-1).t()
        )

        negative_indices = []

        for i in range(batch_size):
            pos_idx = positive_indices[i]
            pos_sim = similarities[i, pos_idx]

            # 排除正样本，创建负样本候选
            candidate_mask = torch.ones(num_candidates, dtype=torch.bool, device=similarities.device)
            candidate_mask[pos_idx] = False

            neg_similarities = similarities[i][candidate_mask]
            neg_candidate_indices = torch.arange(num_candidates, device=similarities.device)[candidate_mask]

            # 选择难样本：相似度高但仍然是负样本的
            hard_threshold = pos_sim - self.margin
            hard_mask = neg_similarities > hard_threshold

            if hard_mask.sum() > 0:
                # 如果有难样本，按相似度排序选择
                hard_similarities = neg_similarities[hard_mask]
                hard_indices = neg_candidate_indices[hard_mask]
                _, sorted_indices = torch.sort(hard_similarities, descending=True)

                num_hard = min(int(num_negatives * self.hard_ratio), len(hard_indices))
                selected_hard = hard_indices[sorted_indices[:num_hard]]

                # 剩余的随机选择
                remaining = num_negatives - num_hard
                if remaining > 0:
                    easy_mask = ~hard_mask
                    easy_indices = neg_candidate_indices[easy_mask]
                    if len(easy_indices) >= remaining:
                        random_easy = easy_indices[torch.randperm(len(easy_indices))[:remaining]]
                        selected_negatives = torch.cat([selected_hard, random_easy])
                    else:
                        selected_negatives = torch.cat([selected_hard, easy_indices])
                        # 如果还不够，从所有负样本中补充
                        remaining_count = num_negatives - len(selected_negatives)
                        all_neg_indices = neg_candidate_indices[torch.randperm(len(neg_candidate_indices))[:remaining_count]]
                        selected_negatives = torch.cat([selected_negatives, all_neg_indices])
                else:
                    selected_negatives = selected_hard
            else:
                # 没有难样本，随机选择
                selected_negatives = neg_candidate_indices[torch.randperm(len(neg_candidate_indices))[:num_negatives]]

            # 确保选择了正确数量的负样本
            if len(selected_negatives) < num_negatives:
                # 重复采样补足
                padding = torch.randint(0, len(neg_candidate_indices),
                                      (num_negatives - len(selected_negatives),),
                                      device=similarities.device)
                padding_indices = neg_candidate_indices[padding]
                selected_negatives = torch.cat([selected_negatives, padding_indices])
            elif len(selected_negatives) > num_negatives:
                selected_negatives = selected_negatives[:num_negatives]

            negative_indices.append(selected_negatives)

        return torch.stack(negative_indices)


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失"""

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query_features: torch.Tensor,
                positive_features: torch.Tensor,
                negative_features: torch.Tensor) -> torch.Tensor:
        """
        计算InfoNCE损失

        Args:
            query_features: [batch_size, d_model]
            positive_features: [batch_size, d_model]
            negative_features: [batch_size, num_negatives, d_model]

        Returns:
            loss: InfoNCE损失
        """
        batch_size = query_features.size(0)

        # 归一化特征
        query_norm = F.normalize(query_features, p=2, dim=-1)
        positive_norm = F.normalize(positive_features, p=2, dim=-1)
        negative_norm = F.normalize(negative_features, p=2, dim=-1)

        # 计算正样本相似度
        positive_similarities = torch.sum(query_norm * positive_norm, dim=-1) / self.temperature

        # 计算负样本相似度
        negative_similarities = torch.bmm(
            query_norm.unsqueeze(1),  # [batch_size, 1, d_model]
            negative_norm.transpose(1, 2)  # [batch_size, d_model, num_negatives]
        ).squeeze(1) / self.temperature  # [batch_size, num_negatives]

        # 构建logits
        logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_features.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class TransformerRetrievalHead(nn.Module):
    """基于Transformer的检索头"""

    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Position encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(d_model, nhead, dropout)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, features: torch.Tensor, context_features: Optional[torch.Tensor] = None):
        """
        Transformer-based feature enhancement

        Args:
            features: [batch_size, seq_len, d_model] or [batch_size, d_model]
            context_features: [batch_size, context_len, d_model] optional context

        Returns:
            enhanced_features: [batch_size, d_model]
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension

        # Add positional encoding
        features = self.pos_encoding(features.transpose(0, 1)).transpose(0, 1)

        # Transformer encoding
        encoded_features = self.transformer(features)

        # Cross-modal attention if context is provided
        if context_features is not None:
            if context_features.dim() == 2:
                context_features = context_features.unsqueeze(1)
            context_features = self.pos_encoding(context_features.transpose(0, 1)).transpose(0, 1)
            encoded_features = self.cross_attention(encoded_features, context_features, context_features)

        # Global pooling
        pooled_features = encoded_features.mean(dim=1)  # [batch_size, d_model]

        # Output projection
        output_features = self.output_projection(pooled_features)

        return output_features


class MultiModalRetriever(nn.Module):
    """高级多模态检索器"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 模型维度
        d_model = getattr(config, 'retrieval_dim', 512)
        self.d_model = d_model

        # 特征投影到统一维度
        self.image_projection = nn.Sequential(
            nn.Linear(config.fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        self.brep_projection = nn.Sequential(
            nn.Linear(config.fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        self.fused_projection = nn.Sequential(
            nn.Linear(config.fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        # Transformer-based retrieval heads
        self.image_retrieval_head = TransformerRetrievalHead(d_model)
        self.brep_retrieval_head = TransformerRetrievalHead(d_model)
        self.fused_retrieval_head = TransformerRetrievalHead(d_model)

        # Advanced similarity computation
        self.similarity_computer = AdvancedSimilarityComputation(d_model)

        # Hard negative mining
        self.hard_negative_miner = HardNegativeMiner()

        # Multi-modal fusion with attention
        self.modality_attention = MultiheadAttention(d_model, 8, batch_first=True)
        self.modality_weights = nn.Parameter(torch.ones(3) / 3)  # image, brep, fused

        # Final output projection
        self.final_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def project_features(self, fusion_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """项目特征到检索空间并增强"""
        projected = {}

        if 'image_enhanced' in fusion_output:
            img_proj = self.image_projection(fusion_output['image_enhanced'])
            projected['image_enhanced'] = self.image_retrieval_head(img_proj)

        if 'brep_enhanced' in fusion_output:
            brep_proj = self.brep_projection(fusion_output['brep_enhanced'])
            projected['brep_enhanced'] = self.brep_retrieval_head(brep_proj)

        if 'fused_features' in fusion_output:
            fused_proj = self.fused_projection(fusion_output['fused_features'])
            # Use cross-modal context for fused features
            context = None
            if 'image_enhanced' in projected and 'brep_enhanced' in projected:
                context = torch.stack([projected['image_enhanced'], projected['brep_enhanced']], dim=1)
            projected['fused_features'] = self.fused_retrieval_head(fused_proj, context)

        return projected

    def compute_similarities(self, query_features: Dict[str, torch.Tensor],
                           candidate_features: Dict[str, torch.Tensor],
                           modality: str = 'fused') -> torch.Tensor:
        """
        使用高级相似度计算
        """
        if modality == 'image':
            return self.similarity_computer(
                query_features['image_enhanced'],
                candidate_features['image_enhanced']
            )
        elif modality == 'brep':
            return self.similarity_computer(
                query_features['brep_enhanced'],
                candidate_features['brep_enhanced']
            )
        elif modality == 'fused':
            return self.similarity_computer(
                query_features['fused_features'],
                candidate_features['fused_features']
            )
        elif modality == 'multi':
            # 多模态加权融合
            similarities = []
            weights = F.softmax(self.modality_weights, dim=0)

            if 'image_enhanced' in query_features:
                img_sim = self.similarity_computer(
                    query_features['image_enhanced'],
                    candidate_features['image_enhanced']
                )
                similarities.append(weights[0] * img_sim)

            if 'brep_enhanced' in query_features:
                brep_sim = self.similarity_computer(
                    query_features['brep_enhanced'],
                    candidate_features['brep_enhanced']
                )
                similarities.append(weights[1] * brep_sim)

            if 'fused_features' in query_features:
                fused_sim = self.similarity_computer(
                    query_features['fused_features'],
                    candidate_features['fused_features']
                )
                similarities.append(weights[2] * fused_sim)

            final_similarities = sum(similarities)
            return final_similarities
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def retrieve(self, query_features: Dict[str, torch.Tensor],
                candidate_features: Dict[str, torch.Tensor],
                k: int = 10, modality: str = 'fused') -> Dict[str, torch.Tensor]:
        """
        执行检索
        """
        # 投影特征
        query_proj = self.project_features(query_features)
        candidate_proj = self.project_features(candidate_features)

        # 计算相似度
        similarities = self.compute_similarities(query_proj, candidate_proj, modality)

        # 获取top-k
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1)

        return {
            'similarities': top_k_similarities,
            'indices': top_k_indices,
            'scores': F.softmax(top_k_similarities, dim=1)
        }

    def compute_contrastive_loss(self, query_features: Dict[str, torch.Tensor],
                                positive_features: Dict[str, torch.Tensor],
                                negative_features: Dict[str, torch.Tensor],
                                modality: str = 'fused') -> torch.Tensor:
        """
        计算对比学习损失
        """
        # 投影特征
        query_proj = self.project_features(query_features)
        positive_proj = self.project_features(positive_features)
        negative_proj = self.project_features(negative_features)

        # 获取对应模态的特征
        if modality == 'fused':
            query_emb = query_proj['fused_features']
            positive_emb = positive_proj['fused_features']
            negative_emb = negative_proj['fused_features']
        elif modality == 'image':
            query_emb = query_proj['image_enhanced']
            positive_emb = positive_proj['image_enhanced']
            negative_emb = negative_proj['image_enhanced']
        elif modality == 'brep':
            query_emb = query_proj['brep_enhanced']
            positive_emb = positive_proj['brep_enhanced']
            negative_emb = negative_proj['brep_enhanced']
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 计算InfoNCE损失
        infonce_loss = InfoNCELoss()
        loss = infonce_loss(query_emb, positive_emb, negative_emb)

        return loss

    def forward(self, query_fusion_output: Dict[str, torch.Tensor],
                candidate_fusion_output: Dict[str, torch.Tensor],
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播（训练模式）

        Args:
            query_fusion_output: 查询的融合特征输出
            candidate_fusion_output: 候选的融合特征输出
            labels: 标签信息（用于计算损失）

        Returns:
            输出字典
        """
        # 投影特征
        query_proj = self.project_features(query_fusion_output)
        candidate_proj = self.project_features(candidate_fusion_output)

        # 计算多模态相似度
        similarities = self.compute_similarities(query_proj, candidate_proj, 'multi')

        result = {
            'similarities': similarities,
            'query_features': query_proj,
            'candidate_features': candidate_proj,
            'modality_weights': F.softmax(self.modality_weights, dim=0)
        }

        return result


class RetrievalMetrics:
    """检索评估指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.total_queries = 0
        self.recall_at_k = {1: 0, 5: 0, 10: 0, 20: 0}
        self.precision_at_k = {1: 0, 5: 0, 10: 0, 20: 0}
        self.map_scores = []

    def update(self, retrieved_indices: torch.Tensor,
               true_labels: torch.Tensor, query_labels: torch.Tensor):
        """
        更新指标

        Args:
            retrieved_indices: [batch_size, k] 检索到的索引
            true_labels: [num_candidates] 候选集的真实标签
            query_labels: [batch_size] 查询的标签
        """
        batch_size = retrieved_indices.size(0)
        self.total_queries += batch_size

        for i in range(batch_size):
            query_label = query_labels[i].item()
            retrieved_idx = retrieved_indices[i]
            retrieved_labels = true_labels[retrieved_idx]

            # 计算每个查询的相关性
            relevant = (retrieved_labels == query_label).float()

            # 更新Recall@K和Precision@K
            for k in self.recall_at_k.keys():
                if k <= len(retrieved_labels):
                    relevant_at_k = relevant[:k]

                    # Recall@K
                    if relevant.sum() > 0:  # 如果有相关文档
                        recall_k = relevant_at_k.sum() / relevant.sum()
                        self.recall_at_k[k] += recall_k.item()

                    # Precision@K
                    precision_k = relevant_at_k.sum() / k
                    self.precision_at_k[k] += precision_k.item()

            # 计算Average Precision (AP)
            if relevant.sum() > 0:
                ap = self._average_precision(relevant)
                self.map_scores.append(ap)

    def _average_precision(self, relevant: torch.Tensor) -> float:
        """计算单个查询的AP"""
        relevant_indices = torch.where(relevant == 1)[0]
        if len(relevant_indices) == 0:
            return 0.0

        ap = 0.0
        for i, idx in enumerate(relevant_indices):
            precision_at_idx = (i + 1) / (idx + 1)
            ap += precision_at_idx

        return ap / len(relevant_indices)

    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        if self.total_queries == 0:
            return {}

        metrics = {}

        # 计算平均Recall@K
        for k in self.recall_at_k.keys():
            metrics[f'Recall@{k}'] = self.recall_at_k[k] / self.total_queries

        # 计算平均Precision@K
        for k in self.precision_at_k.keys():
            metrics[f'Precision@{k}'] = self.precision_at_k[k] / self.total_queries

        # 计算mAP
        if self.map_scores:
            metrics['mAP'] = sum(self.map_scores) / len(self.map_scores)
        else:
            metrics['mAP'] = 0.0

        return metrics


def create_retrieval_model(config: Optional[ModelConfig] = None):
    """创建检索模型的工厂函数"""
    return MultiModalRetriever(config)


def create_retrieval_loss(temperature: float = 0.07):
    """创建检索损失函数的工厂函数"""
    return InfoNCELoss(temperature)


if __name__ == "__main__":
    # 测试检索模型
    config = ModelConfig()
    retriever = create_retrieval_model(config)
    criterion = create_retrieval_loss()

    # 模拟输入数据
    batch_size = 4
    num_candidates = 100

    # 模拟融合特征输出
    query_fusion_output = {
        'fused_features': torch.randn(batch_size, config.fusion_dim),
        'image_enhanced': torch.randn(batch_size, config.fusion_dim),
        'brep_enhanced': torch.randn(batch_size, config.fusion_dim)
    }

    candidate_fusion_output = {
        'fused_features': torch.randn(num_candidates, config.fusion_dim),
        'image_enhanced': torch.randn(num_candidates, config.fusion_dim),
        'brep_enhanced': torch.randn(num_candidates, config.fusion_dim)
    }

    # 模拟标签（正样本在候选集中的索引）
    labels = torch.randint(0, num_candidates, (batch_size,))

    # 测试前向传播
    with torch.no_grad():
        output = retriever(query_fusion_output, candidate_fusion_output, labels)

        print("检索模型测试结果:")
        print(f"  相似度矩阵形状: {output['similarities'].shape}")
        print(f"  模态权重: {output['modality_weights']}")

        # 测试检索功能
        retrieval_results = retriever.retrieve(query_fusion_output, candidate_fusion_output, k=10)
        print(f"  Top-10检索结果形状: {retrieval_results['indices'].shape}")
        print(f"  Top-10相似度形状: {retrieval_results['similarities'].shape}")

        # 测试损失计算
        if hasattr(criterion, '__call__'):
            if isinstance(criterion, InfoNCELoss):
                # InfoNCE损失需要不同的输入格式
                query_emb = output['query_features']['fused_features']
                pos_emb = output['candidate_features']['fused_features'][labels]
                neg_emb = output['candidate_features']['fused_features'][
                    torch.randint(0, num_candidates, (batch_size, 5))
                ]
                loss_value = criterion(query_emb, pos_emb, neg_emb)
            else:
                loss_output = criterion(output, labels)
                loss_value = loss_output['total_loss']
            print(f"  检索损失: {loss_value.item():.4f}")

        # 测试评估指标
        metrics = RetrievalMetrics()
        candidate_labels = torch.randint(0, config.num_classes, (num_candidates,))
        query_labels = candidate_labels[labels]  # 确保查询有匹配的候选

        metrics.update(retrieval_results['indices'], candidate_labels, query_labels)
        metric_results = metrics.compute()

        print("  评估指标:")
        for metric_name, value in metric_results.items():
            print(f"    {metric_name}: {value:.4f}")

    print("✓ 检索模型测试通过！")