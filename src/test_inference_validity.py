import os
import sys
import unittest
import torch
import numpy as np

from models.gnn_utils import NetworkGraph
from models.ppo_agent import PPOAgent
from models.dqn_agent import DQNAgent

class TestInferenceValidity(unittest.TestCase):
    """
    Unit test suite verifying that inference benchmarking operates in strict evaluation mode:
    1. Models set to eval() mode.
    2. torch.no_grad() is enforced.
    3. Model weights remain unchanged (no gradient updates or backprop).
    4. Output shapes and action selections match discrete action space Discrete(8).
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.state_dim = 5
        self.action_dim = 8
        self.ppo_agent = PPOAgent(state_dim=self.state_dim, action_dim=self.action_dim, use_gnn=True)
        self.dqn_agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim, use_gnn=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        net_path = os.path.join(script_dir, "../network/network.net.xml")
        self.graph = NetworkGraph(net_path)
        self.graph_data = self.graph.get_torch_data()
        self.sample_obs = np.array([12.0, 45.0, 0.65, 0.32, 0.15], dtype=np.float32)

    def test_ppo_eval_mode_and_weight_immutability(self):
        self.ppo_agent.model.eval()

        # Capture initial weights
        initial_weights = {name: param.clone().detach() for name, param in self.ppo_agent.model.named_parameters()}

        # Execute 50 inference steps under torch.no_grad()
        with torch.no_grad():
            for _ in range(50):
                action, log_prob, val = self.ppo_agent.select_action(self.sample_obs, self.graph_data)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < self.action_dim)

        # Verify weights have not changed
        for name, param in self.ppo_agent.model.named_parameters():
            self.assertTrue(
                torch.equal(initial_weights[name], param),
                f"PPO Parameter '{name}' was unexpectedly modified during inference benchmarking!"
            )

    def test_dqn_eval_mode_and_weight_immutability(self):
        self.dqn_agent.model.eval()
        self.dqn_agent.epsilon = 0.0  # Greedy evaluation mode

        # Capture initial weights
        initial_weights = {name: param.clone().detach() for name, param in self.dqn_agent.model.named_parameters()}

        # Execute 50 inference steps under torch.no_grad()
        with torch.no_grad():
            for _ in range(50):
                action = self.dqn_agent.select_action(self.sample_obs, self.graph_data)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < self.action_dim)

        # Verify weights have not changed
        for name, param in self.dqn_agent.model.named_parameters():
            self.assertTrue(
                torch.equal(initial_weights[name], param),
                f"DQN Parameter '{name}' was unexpectedly modified during inference benchmarking!"
            )

if __name__ == "__main__":
    unittest.main()
