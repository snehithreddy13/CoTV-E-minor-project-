import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleGCN(nn.Module):
    """
    A simple Graph Convolutional Network layer for embedding the road network.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SimpleGCN, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Basic GCN: x' = AXW (normalized A not implemented here for simplicity)
        # adj should be (N, N)
        out = torch.matmul(adj, x)
        out = self.lin(out)
        return F.relu(out)

class GNNEncoder(nn.Module):
    """
    GNN-based encoder that processes the network graph and returns node embeddings.
    """
    def __init__(self, node_features: int, hidden_dim: int, embedding_dim: int) -> None:
        super(GNNEncoder, self).__init__()
        self.conv1 = SimpleGCN(node_features, hidden_dim)
        self.conv2 = SimpleGCN(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return x

class NetworkGraph:
    """
    Parses SUMO .net.xml and provides graph representation.
    Nodes: Junctions
    Edges: Lanes/Edges
    """
    def __init__(self, net_xml_path: str) -> None:
        self.net_xml_path = net_xml_path
        self.junctions: Dict[str, Tuple[float, float]] = {}  # junction_id -> (x, y)
        self.edges: Dict[str, Tuple[str, str, float]] = {}   # edge_id -> (from_junction, to_junction, length)
        self.adj_matrix: Optional[np.ndarray] = None
        self.node_to_idx: Dict[str, int] = {}
        self.edge_to_idx: Dict[str, int] = {}
        self._parse_net()
        self._build_adjacency()

    def _parse_net(self) -> None:
        tree = ET.parse(self.net_xml_path)
        root = tree.getroot()

        for junction in root.findall('junction'):
            if junction.get('type') != 'internal':
                jid = str(junction.get('id', ''))
                x = float(junction.get('x', 0.0))
                y = float(junction.get('y', 0.0))
                if jid:
                    self.junctions[jid] = (x, y)

        self.node_to_idx = {jid: i for i, jid in enumerate(self.junctions.keys())}
        
        for edge in root.findall('edge'):
            if 'from' in edge.attrib and 'to' in edge.attrib:
                eid = str(edge.get('id', ''))
                from_node = str(edge.get('from', ''))
                to_node = str(edge.get('to', ''))
                # Only include edges between valid junctions we parsed
                if from_node in self.junctions and to_node in self.junctions and eid:
                    # length is usually in lane
                    lanes = edge.findall('lane')
                    length = float(lanes[0].get('length', 100.0)) if lanes else 100.0
                    self.edges[eid] = (from_node, to_node, length)

        self.edge_to_idx = {eid: i for i, eid in enumerate(self.edges.keys())}

    def _build_adjacency(self) -> None:
        num_nodes = len(self.junctions)
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for _, (u, v, _) in self.edges.items():
            u_idx = self.node_to_idx.get(u)
            v_idx = self.node_to_idx.get(v)
            if u_idx is not None and v_idx is not None:
                # Simple adjacency: 1 if connected
                self.adj_matrix[u_idx, v_idx] = 1.0 # Directed graph

    def get_torch_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to torch tensors
        if self.adj_matrix is None:
            self._build_adjacency()
        
        adj: torch.Tensor = torch.from_numpy(self.adj_matrix)  # type: ignore
        
        # Simple node features: normalized coordinates
        coords = np.array(list(self.junctions.values()), dtype=np.float32)
        if coords.shape[0] > 0:
            mean = coords.mean(axis=0)
            std = coords.std(axis=0) + 1e-6
            coords = (coords - mean) / std
            
        x = torch.from_numpy(coords)
        return x, adj
