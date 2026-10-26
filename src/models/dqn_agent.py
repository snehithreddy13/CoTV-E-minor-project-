import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .gnn_utils import GNNEncoder

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, node_features: int = 2, embed_dim: int = 16, use_gnn: bool = True) -> None:
        super(DQNNetwork, self).__init__()
        self.use_gnn = use_gnn
        if self.use_gnn:
            self.gnn = GNNEncoder(node_features, 32, embed_dim)
            self.fc1 = nn.Linear(state_dim + embed_dim, 64)
        else:
            self.fc1 = nn.Linear(state_dim, 64)
            
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state: torch.Tensor, graph_x: Optional[torch.Tensor] = None, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_gnn:
            if graph_x is not None and adj is not None:
                node_idx = state[:, 0].long()
                node_embeds = self.gnn(graph_x, adj)
                # Ensure index is within range [0, num_nodes - 1]
                node_idx = torch.clamp(node_idx, 0, node_embeds.shape[0] - 1)
                current_embed = node_embeds[node_idx]
                x = torch.cat([state, current_embed], dim=1)
            else:
                # Fallback: Zero embedding if graph data is missing but GNN is expected
                zero_embed = torch.zeros((state.shape[0], self.gnn.conv2.lin.out_features), device=state.device)
                x = torch.cat([state, zero_embed], dim=1)
        else:
            x = state
            
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64, use_gnn: bool = True, epsilon: float = 1.0) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_dim, action_dim, use_gnn=use_gnn).to(self.device)
        self.target_model = DQNNetwork(state_dim, action_dim, use_gnn=use_gnn).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory: deque = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray, graph_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 7) # Discrete(8)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x, adj = graph_data if graph_data else (None, None)
        
        with torch.no_grad():
            q_values = self.model(state_t, 
                                  x.to(self.device) if x is not None else None, 
                                  adj.to(self.device) if adj is not None else None)
            
        return int(torch.argmax(q_values).item())

    def update(self) -> None:
        if len(self.memory) < self.batch_size:
            return
            
        # Simplified update
        pass
