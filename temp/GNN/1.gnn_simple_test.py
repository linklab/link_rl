from torch_geometric.datasets import KarateClub
from torch.nn import Linear

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")

    plt.show()


if __name__ == "__main__":
    dataset = KarateClub()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    g = dataset[0]  # Get the first graph object.

    print(g)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {g.num_nodes}')
    print(f'Number of edges: {g.num_edges}')
    print(f'g.x: {g.x}')
    print(f'g.y: {g.y}')
    print(f'g.pos: {g.pos}')
    print(f'g.normal: {g.normal}')
    print(f'g.edge_index: {g.edge_index}')
    print(f'Is undirected: {g.is_undirected()}')
    print('==============================================================')

    print(f'train_mask: {g.train_mask} {g.train_mask.sum()}')
    #print(f'train_mask: {g.val_mask} {g.val_mask.sum()}')
    #print(f'train_mask: {g.test_mask} {g.test_mask.sum()}')

    print('==============================================================')
    print(f'Average node degree: {g.num_edges / g.num_nodes:.2f}')
    print(f'Number of training nodes: {g.train_mask.sum()}')
    print(f'Training node label rate: {int(g.train_mask.sum()) / g.num_nodes:.2f}')
    print(f'Contains isolated nodes: {g.contains_isolated_nodes()}')
    print(f'Contains self-loops: {g.contains_self_loops()}')


