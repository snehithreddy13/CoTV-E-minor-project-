import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from traci.exceptions import FatalTraCIError
from typing import Dict, Any, Tuple, List, Optional

class AmbulanceEnv(gym.Env):
    """
    Custom Environment for Ambulance Routing that securely matches the standard Gymnasium architecture.
    """
    def __init__(self, ambulance_ids: List[str], network_graph: Optional[Any] = None) -> None:
        super(AmbulanceEnv, self).__init__()
        self.ambulance_ids = ambulance_ids
        self.graph = network_graph
        
        # Action space: Select one of the available next edges at current junction
        # For simplicity, we assume a max of 8 possible next edges
        self.action_space = spaces.Discrete(8)
        
        # Observation space: [Current Edge Idx, Target Edge Idx, Speed, Dist to Target, Traffic Density]
        # Using a Box space for features
        num_features = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        self.state = {} # Per ambulance state

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Restores tracking state. Native SUMO resets are globally handled by Traci start parameters."""
        super().reset(seed=seed)
        obs = self._get_obs(self.ambulance_ids[0]) if self.ambulance_ids else np.zeros(5, dtype=np.float32)
        return obs, {}

    def step(self, action_dict: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Executes parallel RL network routing actions sequentially across tracked ambulances.
        
        Args:
            action_dict: A dictionary mapping ambulance_id strings to discrete action integers.
        """
        rewards: Dict[str, float] = {}
        terminated: Dict[str, bool] = {}
        truncated: Dict[str, bool] = {}
        observations: Dict[str, np.ndarray] = {}
        
        for amb_id, action in action_dict.items():
            if amb_id not in traci.vehicle.getIDList():
                continue
                
            # Execute action: Force path to a specific next edge
            # This is complex in TraCI, usually done via traci.vehicle.setRoute
            # For now, we'll map action to 'next edge choice' if the vehicle is at a junction
            self._apply_action(amb_id, action)
            
            # Calculate reward
            rewards[amb_id] = self._calculate_reward(amb_id)
            observations[amb_id] = self._get_obs(amb_id)
            terminated[amb_id] = False # Handled by simulation loop
            truncated[amb_id] = False
            
        return observations, rewards, terminated, truncated, {}

    def _get_obs(self, amb_id: str) -> np.ndarray:
        """
        Standardized 5D Observation for GNN:
        1. Current Node Index (int - for GNN indexing)
        2. Target Node Index (int - for spatial guidance)
        3. Speed (0-1)
        4. Distance to Target (0-1)
        5. Traffic Density (0-1)
        """
        try:
            current_edge = traci.vehicle.getRoadID(amb_id)
            speed = traci.vehicle.getSpeed(amb_id)
            
            # Identify active target (Phase 1 vs Phase 2)
            target = traci.vehicle.getParameter(amb_id, "mission_target")
            if not target:
                target = traci.vehicle.getParameter(amb_id, "return_target")
            
            dist_to_target = traci.vehicle.getDrivingDistance(amb_id, target, 0)
            if dist_to_target < 0: dist_to_target = 1000.0
            
            # Map edge IDs to graph node indices for the GNN
            node_idx = 0
            target_idx = 0
            if self.graph:
                if current_edge in self.graph.edges:
                    node_idx = self.graph.node_to_idx.get(self.graph.edges[current_edge][1], 0)
                if target in self.graph.edges:
                    target_idx = self.graph.node_to_idx.get(self.graph.edges[target][1], 0)
            
            traffic_density = len(traci.vehicle.getIDList()) / 100.0
            
            return np.array([
                float(node_idx), 
                float(target_idx), 
                speed / 20.0, 
                dist_to_target / 1000.0, 
                traffic_density
            ], dtype=np.float32)
        except:
            return np.zeros(5, dtype=np.float32)

    def _apply_action(self, amb_id: str, action: int) -> None:
        """
        Unified Two-Phase Rerouting:
        Uses native SUMO targets to prevent route corruption.
        """
        try:
            if amb_id not in traci.vehicle.getIDList():
                return
                
            current_edge = traci.vehicle.getRoadID(amb_id)
            if current_edge.startswith(':'):
                return
            
            # 1. Identify valid next turns
            lane_id = traci.vehicle.getLaneID(amb_id)
            links = traci.lane.getLinks(lane_id)
            available = [traci.lane.getEdgeID(l[0]) for l in links if not traci.lane.getEdgeID(l[0]).startswith(':')]
            if not available: return
                
            # 2. Select model's choice
            target_next_edge = available[action % len(available)]
            
            # 3. Identify CURRENT active target (Phase 1 or Phase 2)
            target = traci.vehicle.getParameter(amb_id, "mission_target")
            if not target:
                target = traci.vehicle.getParameter(amb_id, "return_target")
            
            # 4. Standardized segment-based reroute
            if target:
                try:
                    route = traci.simulation.findRoute(target_next_edge, target, vType="ambulance")
                    if route.edges:
                        # CRITICAL: Preserve the return journey if we are still in dispatch phase!
                        mission_target = traci.vehicle.getParameter(amb_id, "mission_target")
                        if mission_target:
                            return_target = traci.vehicle.getParameter(amb_id, "return_target")
                            route_back = traci.simulation.findRoute(mission_target, return_target, vType="ambulance")
                            full_edges = [current_edge] + list(route.edges) + list(route_back.edges)[1:]
                            traci.vehicle.setRoute(amb_id, full_edges)
                        else:
                            # Already in return phase, just update route to hospital
                            traci.vehicle.setRoute(amb_id, [current_edge] + list(route.edges))
                except traci.TraCIException:
                    pass
        except:
            pass

    def _calculate_reward(self, amb_id: str) -> float:
        """
        Sophisticated Reward Calculation aligned with cotv 12:
        - Penalizes global vehicle waiting time.
        - HIGH guidance for distance to target (Primary objective).
        - Speed bonuses for ambulance movement.
        - Massive bonus for successful mission completion.
        """
        reward = 0
        
        try:
            current_vehicles = traci.vehicle.getIDList()
            if amb_id not in current_vehicles:
                return 0.0
                
            # 1. Global Traffic Penalty (Mild)
            all_vehicles = current_vehicles
            if all_vehicles:
                total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in all_vehicles)
                avg_waiting = total_waiting / len(all_vehicles)
                reward -= 0.1 * avg_waiting
            
            # 2. Distance-to-Target Guidance (Strategic Objective)
            mission_target = traci.vehicle.getParameter(amb_id, "mission_target")
            if not mission_target:
                mission_target = traci.vehicle.getParameter(amb_id, "return_target")
            
            if not mission_target:
                # Absolute fallback
                full_v_route = traci.vehicle.getRoute(amb_id)
                mission_target = full_v_route[-1] if full_v_route else ""
            
            if mission_target:
                try:
                    dist = traci.vehicle.getDrivingDistance(amb_id, mission_target, 0)
                    if dist >= 0:
                        # Massive guidance: Incentivize closing the gap
                        reward -= (dist / 2.0) 
                        
                        # Bonus: Close proximity 'Arrival' Reward
                        if dist < 10.0:
                            reward += 500.0
                except:
                    pass
            
            # 3. Ambulance Specific Rewards/Penalties
            # Penalty for wait time (Standard)
            waiting_time = traci.vehicle.getWaitingTime(amb_id)
            reward -= 10.0 * waiting_time
            
            # Bonus for maintaining speed
            speed = traci.vehicle.getSpeed(amb_id)
            reward += 1.0 * speed # Doubled speed bonus
            
            # 4. Collision Penalty
            num_collisions = traci.simulation.getCollidingVehiclesNumber()
            if num_collisions > 0:
                reward -= 1000.0 # Increased penalty
                
        except traci.TraCIException:
            pass
            
        return reward
