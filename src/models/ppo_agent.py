import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Any, Optional
from .gnn_utils import GNNEncoder

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, node_features: int = 2, embed_dim: int = 16, use_gnn: bool = True) -> None:
        super(ActorCritic, self).__init__()
        self.use_gnn = use_gnn
        if self.use_gnn:
            self.gnn = GNNEncoder(node_features, 32, embed_dim)
            combined_dim = state_dim + embed_dim
        else:
            combined_dim = state_dim
            
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, graph_x: Optional[torch.Tensor] = None, adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
            
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, use_gnn: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim, use_gnn=use_gnn).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, graph_data: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x, adj = graph_data if graph_data else (None, None)
        
        with torch.no_grad():
            probs, value = self.model(state_t, 
                                     x.to(self.device) if x is not None else None, 
                                     adj.to(self.device) if adj is not None else None)
            
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
        
    def update(self, rollouts: Any) -> None:
        # Simplified PPO update footprint
        pass
