import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

# Define Graph structure
class Graph:
    def __init__(self):
        """
        初始化图结构类。

        Examples:
        >>> graph = Graph()
        """
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, features):
        """
        添加节点到图中。

        Args:
        - node_id (int): 节点的唯一标识符。
        - features (list): 节点的特征向量。

        Examples:
        >>> graph = Graph()
        >>> graph.add_node(0, [1.0, 0.5])
        """
        self.nodes[node_id] = features

    def add_edge(self, node1, node2):
        """
        添加边到图中。

        Args:
        - node1 (int): 边的一个节点。
        - node2 (int): 边的另一个节点。

        Examples:
        >>> graph = Graph()
        >>> graph.add_edge(0, 1)
        """
        self.edges.append((node1, node2))

    def get_adjacency_matrix(self):
        """
        获取邻接矩阵。

        Returns:
        - adjacency_matrix (numpy.ndarray): 邻接矩阵。

        Examples:
        >>> graph = Graph()
        >>> graph.add_node(0, [1.0, 0.5])
        >>> graph.add_node(1, [0.5, 1.5])
        >>> graph.add_edge(0, 1)
        >>> adjacency_matrix = graph.get_adjacency_matrix()
        """
        num_nodes = len(self.nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for edge in self.edges:
            node1, node2 = edge
            adjacency_matrix[node1, node2] = 1
            adjacency_matrix[node2, node1] = 1  # assuming undirected graph
        return adjacency_matrix

    def get_feature_matrix(self):
        """
        获取特征矩阵。

        Returns:
        - features (numpy.ndarray): 特征矩阵。

        Examples:
        >>> graph = Graph()
        >>> graph.add_node(0, [1.0, 0.5])
        >>> graph.add_node(1, [0.5, 1.5])
        >>> features = graph.get_feature_matrix()
        """
        features = np.array([self.nodes[node] for node in sorted(self.nodes)])
        return features

# Define GCN layers
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, feature_matrix):
        """
        GCN层的前向传播函数。

        Args:
        - adjacency_matrix (torch.Tensor): 邻接矩阵。
        - feature_matrix (torch.Tensor): 特征矩阵。

        Returns:
        - transformed_features (torch.Tensor): 经过GCN层变换后的特征矩阵。

        Examples:
        >>> gcn_layer = GCNLayer(2, 4)
        >>> adjacency_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        >>> feature_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float32)
        >>> transformed_features = gcn_layer(adjacency_matrix, feature_matrix)
        """
        # 计算度矩阵
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        # 计算度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix))
        # 标准化邻接矩阵
        normalized_adjacency_matrix = torch.mm(torch.mm(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt)

        # 聚合特征
        aggregated_features = torch.mm(normalized_adjacency_matrix, feature_matrix)
        # 线性变换
        transformed_features = self.linear(aggregated_features)
        # 使用ReLU激活函数
        return F.relu(transformed_features)

# Define GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        """
        初始化GCN模型。

        Args:
        - num_features (int): 输入特征的维度。
        - hidden_dim (int): 隐藏层的维度。
        - num_classes (int): 输出类别的数量。

        Examples:
        >>> gcn = GCN(2, 4, 2)
        """
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, adjacency_matrix, feature_matrix):
        """
        GCN模型的前向传播函数。

        Args:
        - adjacency_matrix (torch.Tensor): 邻接矩阵。
        - feature_matrix (torch.Tensor): 特征矩阵。

        Returns:
        - output_rep (torch.Tensor): 经过GCN模型变换后的特征矩阵。

        Examples:
        >>> gcn = GCN(2, 4, 2)
        >>> adjacency_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        >>> feature_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float32)
        >>> output_rep = gcn(adjacency_matrix, feature_matrix)
        """
        # 第一层GCN
        hidden_rep = self.gcn1(adjacency_matrix, feature_matrix)
        # 第二层GCN
        output_rep = self.gcn2(adjacency_matrix, hidden_rep)
        return output_rep

# 创建一个示例图
graph = Graph()
graph.add_node(0, [1.0, 0.5])
graph.add_node(1, [0.5, 1.5])
graph.add_node(2, [1.5, 1.0])
graph.add_node(3, [0.0, 1.0])

graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 0)

adjacency_matrix = graph.get_adjacency_matrix()
feature_matrix = graph.get_feature_matrix()
# Define the GCN model
num_features = feature_matrix.shape[1]
hidden_dim = 4
num_classes = 2

gcn = GCN(num_features, hidden_dim, num_classes)

# Convert adjacency matrix and feature matrix to tensors
adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
feature_matrix_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

# Forward pass
output = gcn(adjacency_matrix_tensor, feature_matrix_tensor)

# Visualization function
def plot_graph(graph, feature_matrix, title):
    """
    绘制图形的函数。

    Args:
    - graph (Graph): 图结构对象。
    - feature_matrix (torch.Tensor): 特征矩阵。
    - title (str): 图形的标题。

    Examples:
    >>> graph = Graph()
    >>> graph.add_node(0, [1.0, 0.5])
    >>> graph.add_node(1, [0.5, 1.5])
    >>> feature_matrix = graph.get_feature_matrix()
    >>> plot_graph(graph, feature_matrix, "Initial Graph")
    """
    G = nx.Graph()
    for node_id, features in graph.nodes.items():
        G.add_node(node_id, features=features)
    G.add_edges_from(graph.edges)
    
    pos = nx.spring_layout(G)
    node_features = feature_matrix.detach().numpy()
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_features[:, 0], cmap=plt.cm.Blues, node_size=500, font_color='white')
    plt.title(title)
    plt.savefig(f'figures/{title}.png')
    plt.close()

# Plot the initial graph
plot_graph(graph, feature_matrix_tensor, "Initial Graph")

# Plot the graph after first GCN layer
with torch.no_grad():
    hidden_rep = gcn.gcn1(adjacency_matrix_tensor, feature_matrix_tensor)
plot_graph(graph, hidden_rep, "Graph after GCN Layer 1")

# Plot the final output graph
plot_graph(graph, output, "Graph after GCN Layer 2")

print(output)