import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os
import tarfile

def load_small_edges_and_features(edge_file_path, feature_file_path, num_nodes=100):
    """
    加载包含边和特征的小规模数据集。

    参数：
    - edge_file_path：边文件的路径
    - feature_file_path：特征文件的路径
    - num_nodes：节点数量，默认为100

    返回值：
    - adjacency_matrix：邻接矩阵
    - feature_matrix：特征矩阵
    """
    edges = []
    with open(edge_file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            if node1 < num_nodes and node2 < num_nodes:
                edges.append((node1, node2))
    
    features = []
    with open(feature_file_path, 'r') as f:
        for line in f:
            features.append(list(map(float, line.strip().split())))
    
    feature_matrix = np.array(features[:num_nodes])
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for node1, node2 in edges:
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # assuming undirected graph
    
    return adjacency_matrix, feature_matrix


# Step 2: 定义GCN层和GCN模型
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, feature_matrix):
        """
        GCN层的前向传播函数。

        参数：
        - adjacency_matrix：邻接矩阵
        - feature_matrix：特征矩阵

        返回值：
        - transformed_features：经过GCN层变换后的特征矩阵
        """
        # Step 2.1: 计算度矩阵
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        # Step 2.2: 计算度矩阵的逆平方根
        degree_matrix_inv_sqrt = torch.inverse(torch.sqrt(degree_matrix) + torch.eye(degree_matrix.size(0)) * 1e-10)
        # Step 2.3: 计算归一化邻接矩阵
        normalized_adjacency_matrix = torch.mm(torch.mm(degree_matrix_inv_sqrt, adjacency_matrix), degree_matrix_inv_sqrt)

        # Step 2.4: 聚合特征
        aggregated_features = torch.mm(normalized_adjacency_matrix, feature_matrix)
        # Step 2.5: 线性变换
        transformed_features = self.linear(aggregated_features)
        return F.relu(transformed_features)

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        """
        GCN模型的初始化函数。

        参数：
        - num_features：输入特征的维度
        - hidden_dim：隐藏层的维度
        - num_classes：输出类别的数量
        """
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, adjacency_matrix, feature_matrix):
        """
        GCN模型的前向传播函数。

        参数：
        - adjacency_matrix：邻接矩阵
        - feature_matrix：特征矩阵

        返回值：
        - output_rep：经过GCN模型变换后的输出特征矩阵
        """
        hidden_rep = self.gcn1(adjacency_matrix, feature_matrix)
        output_rep = self.gcn2(adjacency_matrix, hidden_rep)
        return output_rep

# Visualization function
def plot_graph(adjacency_matrix, feature_matrix, title):
    """
    绘制图形的函数。

    参数：
    - adjacency_matrix：邻接矩阵
    - feature_matrix：特征矩阵
    - title：图形标题

    返回值：无
    """
    G = nx.Graph()
    edges = np.argwhere(adjacency_matrix > 0)
    edges = [(int(node1), int(node2)) for node1, node2 in edges]
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    node_features = feature_matrix.detach().numpy()
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_features[:, 0], cmap=plt.cm.Blues, node_size=500, font_color='white')
    plt.title(title)
    plt.savefig(f'figures/{title}.png')
    plt.close()
    
if __name__ == '__main__':
    # Step 1: 解压提供的 tar.gz 文件
    file_path = 'dataset/facebook/facebook.tar.gz'
    extracted_dir = 'dataset/facebook'

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)

    # 加载解压后目录的内容
    facebook_dir = os.path.join(extracted_dir, 'facebook')
    # 选择前100个节点进行处理
    adjacency_matrix, feature_matrix = load_small_edges_and_features(os.path.join(facebook_dir, '0.edges'), os.path.join(facebook_dir, '0.feat'), num_nodes=100)
    # 将邻接矩阵和特征矩阵转换为张量
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    feature_matrix_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
    # 定义 GCN 模型
    num_features = feature_matrix_tensor.shape[1]
    hidden_dim = 16
    num_classes = 2  # 为简单起见，我们假设是二分类问题
    gcn = GCN(num_features, hidden_dim, num_classes)
    # 前向传播
    output = gcn(adjacency_matrix_tensor, feature_matrix_tensor)
    print(output)
    # 绘制初始图
    plot_graph(adjacency_matrix, feature_matrix_tensor, "初始图")
    # 绘制经过第一层 GCN 后的图
    with torch.no_grad():
        hidden_rep = gcn.gcn1(adjacency_matrix_tensor, feature_matrix_tensor)
    plot_graph(adjacency_matrix, hidden_rep, "经过第一层 GCN 后的图")
    # 绘制最终输出图
    plot_graph(adjacency_matrix, output, "经过第二层 GCN 后的图")