"""
Hospital Selector Engine for CoTV-E Simulation
Implements a multi-criteria decision dispatch policy evaluating travel time,
hospital capacity, trauma center level, and operational availability.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
import traci

class HospitalSelector:
    """
    Transparent Multi-Criteria Hospital Selector for Emergency Dispatches.
    Evaluates candidate hospitals using live traffic, network distance,
    ICU bed availability, trauma center capabilities, and operational status.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.hospitals: List[Dict[str, Any]] = []
        self.load_config()

    def load_config(self) -> None:
        """Loads hospital profiles from hospitals_config.json."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Hospital configuration file not found at: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.hospitals = data.get('hospitals', [])

    def evaluate_hospitals(
        self,
        emergency_edge: str,
        emergency_severity: str = "URGENT",
        vehicle_type: str = "ambulance"
    ) -> Dict[str, Any]:
        """
        Evaluates all candidate hospitals for an emergency location.

        Args:
            emergency_edge: SUMO edge ID where emergency occurred.
            emergency_severity: 'CRITICAL_TRAUMA', 'URGENT', or 'STABLE'.
            vehicle_type: SUMO vType used for route travel time calculations.

        Returns:
            Dictionary with candidate breakdown, selected hospital, and selection reason.
        """
        candidates: List[Dict[str, Any]] = []
        best_candidate: Optional[Dict[str, Any]] = None
        min_score = float('inf')

        for hosp in self.hospitals:
            hosp_id = hosp.get('id', 'UNKNOWN')
            hosp_name = hosp.get('name', 'Hospital')
            capacity = hosp.get('capacity', {})
            status = capacity.get('status', 'OPERATIONAL')
            icu_avail = capacity.get('icu_beds_available', 0)
            icu_total = capacity.get('icu_beds_total', 1)
            trauma_level = capacity.get('trauma_center_level', 2)

            # 1. Availability Check
            if status != 'OPERATIONAL':
                candidates.append({
                    'id': hosp_id,
                    'name': hosp_name,
                    'available': False,
                    'reason': f"Hospital status is {status}",
                    'score': float('inf')
                })
                continue

            if icu_avail <= 0:
                candidates.append({
                    'id': hosp_id,
                    'name': hosp_name,
                    'available': False,
                    'reason': "Zero ICU beds available (Capacity Exhausted)",
                    'score': float('inf')
                })
                continue

            # Pick representative entrance and return edges
            access_edges = hosp.get('access_edges', [])
            return_edges = hosp.get('return_edges', [])

            if not access_edges or not return_edges:
                candidates.append({
                    'id': hosp_id,
                    'name': hosp_name,
                    'available': False,
                    'reason': "Missing network access/return edges",
                    'score': float('inf')
                })
                continue

            # Pick best access/return edge pair using SUMO findRoute travel time
            best_outbound_time = float('inf')
            best_return_time = float('inf')
            chosen_start_edge = access_edges[0]
            chosen_return_edge = return_edges[0]

            for s_edge in access_edges:
                try:
                    ro = traci.simulation.findRoute(s_edge, emergency_edge, vType=vehicle_type)
                    if ro.edges and ro.travelTime > 0 and ro.travelTime < best_outbound_time:
                        best_outbound_time = ro.travelTime
                        chosen_start_edge = s_edge
                except Exception:
                    pass

            for r_edge in return_edges:
                try:
                    rb = traci.simulation.findRoute(emergency_edge, r_edge, vType=vehicle_type)
                    if rb.edges and rb.travelTime > 0 and rb.travelTime < best_return_time:
                        best_return_time = rb.travelTime
                        chosen_return_edge = r_edge
                except Exception:
                    pass

            if best_outbound_time == float('inf') or best_return_time == float('inf'):
                candidates.append({
                    'id': hosp_id,
                    'name': hosp_name,
                    'available': False,
                    'reason': "No valid route found in current network state",
                    'score': float('inf')
                })
                continue

            total_travel_time = best_outbound_time + best_return_time

            # 2. Capacity Load Ratio (0 to 100%)
            occupancy_ratio = (icu_total - icu_avail) / max(icu_total, 1)
            capacity_cost = occupancy_ratio * 50.0  # max 50 points penalty for full load

            # 3. Trauma Level Match Penalty
            trauma_penalty = 0.0
            if emergency_severity == "CRITICAL_TRAUMA" and trauma_level > 1:
                trauma_penalty = 45.0  # 45s equivalent cost penalty for non-Level 1 trauma center

            # 4. Multi-Criteria Score Calculation (Cost Minimization)
            # Score = 0.65 * travel_time + capacity_cost + trauma_penalty
            composite_score = (0.65 * total_travel_time) + capacity_cost + trauma_penalty

            candidate_info = {
                'id': hosp_id,
                'name': hosp_name,
                'available': True,
                'start_edge': chosen_start_edge,
                'return_edge': chosen_return_edge,
                'outbound_time': round(best_outbound_time, 1),
                'return_time': round(best_return_time, 1),
                'total_travel_time': round(total_travel_time, 1),
                'icu_beds_available': icu_avail,
                'icu_beds_total': icu_total,
                'trauma_level': trauma_level,
                'score': round(composite_score, 2)
            }
            candidates.append(candidate_info)

            if composite_score < min_score:
                min_score = composite_score
                best_candidate = candidate_info

        if not best_candidate:
            # Fallback to first available if all penalized
            valid_candidates = [c for c in candidates if c.get('available')]
            if valid_candidates:
                best_candidate = valid_candidates[0]
            else:
                raise RuntimeError("No candidate hospitals available for dispatch!")

        # Formulate selection rationale
        reasons = []
        reasons.append(f"Fastest total travel time ({best_candidate['total_travel_time']}s)")
        reasons.append(f"ICU Capacity: {best_candidate['icu_beds_available']}/{best_candidate['icu_beds_total']} beds free")
        if emergency_severity == "CRITICAL_TRAUMA":
            reasons.append(f"Level {best_candidate['trauma_level']} Trauma Center match")

        selection_reason = ", ".join(reasons)

        return {
            'selected_hospital_id': best_candidate['id'],
            'selected_hospital_name': best_candidate['name'],
            'start_edge': best_candidate['start_edge'],
            'return_edge': best_candidate['return_edge'],
            'est_travel_time': best_candidate['total_travel_time'],
            'selection_reason': selection_reason,
            'decision_score': best_candidate['score'],
            'candidates': candidates
        }
