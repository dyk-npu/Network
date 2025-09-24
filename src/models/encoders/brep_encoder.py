import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import logging
from typing import Dict, Any, Optional
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """图卷积层"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = dglnn.GraphConv(in_dim, out_dim, activation=F.relu)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        x = self.conv(graph, features)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class GraphAttentionLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = dglnn.GATConv(
            in_dim, out_dim, num_heads, feat_drop=dropout, attn_drop=dropout,
            activation=F.relu, allow_zero_in_degree=True
        )
        self.norm = nn.LayerNorm(out_dim * num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        x = self.attention(graph, features)
        x = x.flatten(1)  # Flatten multi-head outputs
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BrepEncoder(nn.Module):
    """B-rep几何编码器

    将B-rep几何图转换为固定维度的特征向量
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 输入特征维度 (点坐标3 + 法向量3 + 可见性1 = 7 for faces)
        # (点坐标3 + 切向量3 = 6 for edges)
        self.face_input_dim = 7
        self.edge_input_dim = 6

        # 面特征编码器
        self.face_encoder = nn.Sequential(
            nn.Linear(self.face_input_dim, config.brep_hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(config.brep_hidden_dims[0]),
            nn.Dropout(config.dropout)
        )

        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_input_dim, config.brep_hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(config.brep_hidden_dims[0]),
            nn.Dropout(config.dropout)
        )

        # 图卷积层
        self.graph_layers = nn.ModuleList()
        in_dim = config.brep_hidden_dims[0]

        for out_dim in config.brep_hidden_dims[1:]:
            layer = GraphAttentionLayer(in_dim, out_dim // 4, 4, config.dropout)
            self.graph_layers.append(layer)
            in_dim = out_dim

        # 全局池化和最终编码 - 使用readout函数确保批次维度正确
        self.global_pool = dglnn.GlobalAttentionPooling(
            nn.Linear(config.brep_hidden_dims[-1], 1)
        )

        # 最终输出层
        self.output_projection = nn.Sequential(
            nn.Linear(config.brep_hidden_dims[-1], config.brep_dim),
            nn.ReLU(),
            nn.LayerNorm(config.brep_dim),
            nn.Dropout(config.dropout)
        )

        # 自适应维度调整层（动态创建）
        self.adaptive_layers = {}

    def _get_adaptive_projection(self, input_dim: int, output_dim: int, device: torch.device) -> nn.Module:
        """获取或创建自适应投影层"""
        key = f"{input_dim}_{output_dim}"
        if key not in self.adaptive_layers:
            layer = nn.Linear(input_dim, output_dim).to(device)
            self.adaptive_layers[key] = layer
            # 将层注册为模块的参数
            self.add_module(f"adaptive_{key}", layer)
        return self.adaptive_layers[key]

    def forward(self, graph: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            graph: DGL图，包含节点特征(面)和边特征

        Returns:
            Dict包含:
                - features: 全局B-rep特征 [batch_size, brep_dim]
                - node_features: 节点级特征 [num_nodes, hidden_dim]
                - attention_weights: 注意力权重(如果可用)
        """
        # 记录图信息用于调试
        logger.debug(f"B-rep encoder input - batch_size: {graph.batch_size}, graphs: {graph.batch_num_nodes().tolist()}")

        # 获取输入特征
        face_features = graph.ndata['x']  # 期望: [num_faces, face_input_dim]

        # 标准化面特征形状处理
        original_shape = face_features.shape
        if len(original_shape) == 4:
            # [num_faces, num_u_samples, num_v_samples, channels]
            num_faces, num_u, num_v, channels = original_shape
            # 如果特征维度不匹配，需要调整到期望维度
            total_features = num_u * num_v * channels
            if total_features != self.face_input_dim:
                # 使用自适应层来调整维度
                face_features = face_features.view(num_faces, total_features)
                face_features = self._get_adaptive_projection(total_features, self.face_input_dim, face_features.device)(face_features)
            else:
                face_features = face_features.view(num_faces, self.face_input_dim)
        elif len(original_shape) == 3:
            # [num_faces, num_samples, channels]
            num_faces, num_samples, channels = original_shape
            total_features = num_samples * channels
            if total_features != self.face_input_dim:
                face_features = face_features.view(num_faces, total_features)
                face_features = self._get_adaptive_projection(total_features, self.face_input_dim, face_features.device)(face_features)
            else:
                face_features = face_features.view(num_faces, self.face_input_dim)
        elif len(original_shape) == 2:
            # [num_faces, feature_dim] - 已经是正确格式
            if original_shape[1] != self.face_input_dim:
                face_features = self._get_adaptive_projection(original_shape[1], self.face_input_dim, face_features.device)(face_features)
        else:
            raise ValueError(f"Unsupported face features shape: {original_shape}")

        # 编码面特征
        node_features = self.face_encoder(face_features)

        # 确保图有自环，避免0入度节点问题
        # 检查是否存在自环
        has_self_loops = False
        try:
            nodes = torch.arange(graph.number_of_nodes(), device=graph.device)
            has_self_loops = graph.has_edges_between(nodes, nodes).any().item()
        except:
            has_self_loops = False

        if not has_self_loops:
            logger.debug("Adding self-loops to maintain batch structure")
            # 手动为每个图添加自环，保持批次结构
            graphs = dgl.unbatch(graph)
            graphs_with_loops = []
            for g in graphs:
                # 检查每个图是否有自环 - 确保节点索引在正确的设备上
                nodes = torch.arange(g.number_of_nodes(), device=g.device)
                has_self_loop = g.has_edges_between(nodes, nodes).any().item() if g.number_of_nodes() > 0 else False
                if not has_self_loop:
                    g = dgl.add_self_loop(g)
                graphs_with_loops.append(g)
            graph = dgl.batch(graphs_with_loops)
            logger.debug(f"After self-loops - batch_size: {graph.batch_size}, graphs: {graph.batch_num_nodes().tolist()}")

        # 处理边特征(如果存在)
        if 'x' in graph.edata and graph.number_of_edges() > 0:
            edge_features = graph.edata['x']
            edge_original_shape = edge_features.shape

            # 标准化边特征形状
            if len(edge_original_shape) == 3:
                num_edges, num_samples, channels = edge_original_shape
                total_edge_features = num_samples * channels
                if total_edge_features != self.edge_input_dim:
                    edge_features = edge_features.view(num_edges, total_edge_features)
                    edge_features = self._get_adaptive_projection(total_edge_features, self.edge_input_dim, edge_features.device)(edge_features)
                else:
                    edge_features = edge_features.view(num_edges, self.edge_input_dim)
            elif len(edge_original_shape) == 2:
                if edge_original_shape[1] != self.edge_input_dim:
                    edge_features = self._get_adaptive_projection(edge_original_shape[1], self.edge_input_dim, edge_features.device)(edge_features)

            # 将边特征投影到相同维度
            edge_encoded = self.edge_encoder(edge_features)

            # 将边特征聚合到节点
            graph.edata['encoded'] = edge_encoded
            graph.update_all(dgl.function.copy_e('encoded', 'm'),
                           dgl.function.mean('m', 'edge_agg'))
            if 'edge_agg' in graph.ndata:
                node_features = node_features + graph.ndata['edge_agg']

        # 通过图卷积层
        logger.debug(f"Before graph convs - batch_size: {graph.batch_size}")
        for i, layer in enumerate(self.graph_layers):
            node_features = layer(graph, node_features)
        logger.debug(f"After graph convs - batch_size: {graph.batch_size}, graphs: {graph.batch_num_nodes().tolist()}")

        # 全局池化
        global_features = self.global_pool(graph, node_features)

        # 最终投影
        output_features = self.output_projection(global_features)

        return {
            'features': output_features,
            'node_features': node_features,
            'global_pooled': global_features
        }

    def get_attention_weights(self, graph: dgl.DGLGraph) -> Dict[str, torch.Tensor]:
        """获取图注意力层的注意力权重"""
        attention_weights = {}

        # 获取输入特征并进行预处理（与forward方法相同）
        face_features = graph.ndata['x']
        original_shape = face_features.shape

        if len(original_shape) == 4:
            num_faces, num_u, num_v, channels = original_shape
            total_features = num_u * num_v * channels
            if total_features != self.face_input_dim:
                face_features = face_features.view(num_faces, total_features)
                face_features = self._get_adaptive_projection(total_features, self.face_input_dim, face_features.device)(face_features)
            else:
                face_features = face_features.view(num_faces, self.face_input_dim)
        elif len(original_shape) == 3:
            num_faces, num_samples, channels = original_shape
            total_features = num_samples * channels
            if total_features != self.face_input_dim:
                face_features = face_features.view(num_faces, total_features)
                face_features = self._get_adaptive_projection(total_features, self.face_input_dim, face_features.device)(face_features)
            else:
                face_features = face_features.view(num_faces, self.face_input_dim)
        elif len(original_shape) == 2:
            if original_shape[1] != self.face_input_dim:
                face_features = self._get_adaptive_projection(original_shape[1], self.face_input_dim, face_features.device)(face_features)

        node_features = self.face_encoder(face_features)

        # 确保图有自环 - 使用保持批次结构的方法
        has_self_loops = False
        try:
            nodes = torch.arange(graph.number_of_nodes(), device=graph.device)
            has_self_loops = graph.has_edges_between(nodes, nodes).any().item()
        except:
            has_self_loops = False

        if not has_self_loops:
            # 手动为每个图添加自环，保持批次结构
            graphs = dgl.unbatch(graph)
            graphs_with_loops = []
            for g in graphs:
                # 检查每个图是否有自环 - 确保节点索引在正确的设备上
                nodes = torch.arange(g.number_of_nodes(), device=g.device)
                has_self_loop = g.has_edges_between(nodes, nodes).any().item() if g.number_of_nodes() > 0 else False
                if not has_self_loop:
                    g = dgl.add_self_loop(g)
                graphs_with_loops.append(g)
            graph = dgl.batch(graphs_with_loops)

        # 逐层提取注意力权重
        for i, layer in enumerate(self.graph_layers):
            if hasattr(layer.attention, 'get_attention'):
                # 如果GAT层支持返回注意力权重
                with torch.no_grad():
                    _, attn_weights = layer.attention(graph, node_features, get_attention=True)
                    attention_weights[f'layer_{i}'] = attn_weights

            node_features = layer(graph, node_features)

        return attention_weights

    def compute_graph_embedding(self, preprocessed_graphs: list) -> torch.Tensor:
        """批量计算图嵌入

        Args:
            preprocessed_graphs: 预处理后的DGL图列表

        Returns:
            torch.Tensor: 批量图嵌入 [batch_size, brep_dim]
        """
        if not preprocessed_graphs:
            return torch.empty(0, self.config.brep_dim)

        # 批量处理图
        batched_graph = dgl.batch(preprocessed_graphs)
        result = self.forward(batched_graph)

        return result['features']




def create_brep_encoder(config: Optional[ModelConfig] = None) -> BrepEncoder:
    """创建B-rep编码器的工厂函数"""
    return BrepEncoder(config)


def validate_brep_encoder(config: Optional[ModelConfig] = None) -> bool:
    """验证B-rep编码器的功能性"""
    try:
        if config is None:
            config = ModelConfig()

        encoder = create_brep_encoder(config)

        # 创建测试图 - 确保连通性
        num_nodes = 6
        src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
        dst = torch.tensor([1, 2, 3, 4, 5, 0, 2, 3, 4])
        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        graph = dgl.add_self_loop(graph)

        # 测试标准输入
        face_features = torch.randn(num_nodes, 7)  # 标准面特征
        graph.ndata['x'] = face_features

        # 前向传播测试
        with torch.no_grad():
            output = encoder(graph)

        # 验证输出形状
        expected_batch_size = 1  # 单个图
        assert output['features'].shape == (expected_batch_size, config.brep_dim)
        assert output['node_features'].shape[0] == num_nodes
        assert output['global_pooled'].shape == (expected_batch_size, config.brep_hidden_dims[-1])

        print("✓ B-rep encoder validation passed!")
        return True

    except Exception as e:
        print(f"✗ B-rep encoder validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_brep_encoder()
    exit(0 if success else 1)