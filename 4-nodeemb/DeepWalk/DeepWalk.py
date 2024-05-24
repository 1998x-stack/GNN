import os
import gzip
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Unzip the gz file and show directory list
gz_file_path = 'dataset/words_dat.txt.gz'
extracted_file_path = 'dataset/words_dat.txt'

# Unzip the gz file
with gzip.open(gz_file_path, 'rb') as f_in:
    with open(extracted_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# List directory contents
extracted_dir = 'dataset'
files = os.listdir(extracted_dir)
print("Files in directory:", files)

# Step 2: Generate graph from the unzipped data
with open(extracted_file_path, 'r') as f:
    lines = f.readlines()

# Generate the graph
graph = nx.Graph()
for line in lines:
    nodes = line.strip().split()
    if len(nodes) == 2:
        graph.add_edge(nodes[0], nodes[1])

# Display some basic information about the graph
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

# Convert the graph to an adjacency list dictionary format for the DeepWalk class
graph_dict = defaultdict(list)
for node in graph.nodes():
    graph_dict[node] = list(graph.neighbors(node))

class DeepWalk:
    """DeepWalk algorithm implemented using PyTorch."""

    def __init__(self, graph: dict, walk_length: int, num_walks: int, embedding_dim: int, window_size: int, lr: float, num_epochs: int):
        """
        Initialize the DeepWalk model.
        
        Args:
            graph (dict): The input graph as an adjacency list.
            walk_length (int): Length of each random walk.
            num_walks (int): Number of walks per node.
            embedding_dim (int): Dimension of the node embeddings.
            window_size (int): Context window size for Skip-gram model.
            lr (float): Learning rate.
            num_epochs (int): Number of epochs for training.
        """
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.vocab = list(graph.keys())
        self.model = SkipGramModel(len(self.vocab), embedding_dim)
        self.node_to_index = {node: idx for idx, node in enumerate(self.vocab)}

    def generate_random_walks(self) -> list:
        """Generate random walks for each node in the graph."""
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.keys():
                walks.append(self.random_walk(node))
        return walks

    def random_walk(self, start_node: str) -> list:
        """Perform a random walk starting from the given node."""
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            if len(self.graph[cur]) > 0:
                walk.append(random.choice(self.graph[cur]))
            else:
                break
        return walk

    def train(self):
        """Train the DeepWalk model."""
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        walks = self.generate_random_walks()
        data = self.generate_training_data(walks)

        for epoch in range(self.num_epochs):
            total_loss = 0
            for target, context in data:
                optimizer.zero_grad()
                loss = self.model(target, context)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss}")

    def generate_training_data(self, walks: list) -> list:
        """Generate training data for the Skip-gram model from random walks."""
        data = []
        for walk in walks:
            indices = [self.node_to_index[node] for node in walk]
            for i, target in enumerate(indices):
                context = [indices[j] for j in range(max(0, i - self.window_size), min(len(indices), i + self.window_size + 1)) if j != i]
                data.append((target, context))
        return data

    def get_embeddings(self) -> np.ndarray:
        """Get the node embeddings."""
        return self.model.input_embeddings.weight.data.numpy()

class SkipGramModel(nn.Module):
    """Skip-gram model for learning node embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize the Skip-gram model.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique nodes).
            embedding_dim (int): Dimension of the embeddings.
        """
        super(SkipGramModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target: int, context: list) -> torch.Tensor:
        """
        Forward pass of the Skip-gram model.
        
        Args:
            target (int): Index of the target node.
            context (list): List of indices of the context nodes.
        
        Returns:
            torch.Tensor: The loss value.
        """
        target_embedding = self.input_embeddings(torch.tensor([target]))
        context_embeddings = self.output_embeddings(torch.tensor(context))
        scores = torch.matmul(context_embeddings, target_embedding.t()).squeeze()
        log_probs = torch.log(torch.sigmoid(scores)).sum()
        negative_samples = torch.randint(0, self.input_embeddings.num_embeddings, context_embeddings.shape)
        negative_context_embeddings = self.output_embeddings(negative_samples)
        negative_scores = torch.matmul(negative_context_embeddings, target_embedding.t()).squeeze()
        negative_log_probs = torch.log(torch.sigmoid(-negative_scores)).sum()
        return - (log_probs + negative_log_probs)

# Initialize and train the DeepWalk model with the generated graph
deepwalk = DeepWalk(graph_dict, walk_length=10, num_walks=80, embedding_dim=64, window_size=5, lr=0.025, num_epochs=10)
deepwalk.train()

# Get the embeddings
embeddings = deepwalk.get_embeddings()

# Visualize using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plt.figure(figsize=(14, 10))
for i, node in enumerate(graph.nodes()):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], node, fontsize=9)
plt.title("Node Embeddings Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(f'output/DeepWalk_PCA.png')
plt.close()