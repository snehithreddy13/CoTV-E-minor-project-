"""
Unit Test for Hospital Selector Engine
Verifies multi-criteria scoring and hospital selection logic.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add src to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from hospital_selector import HospitalSelector

class TestHospitalSelector(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(src_dir, "../database/hospitals_config.json")
        self.selector = HospitalSelector(self.config_path)

    def test_load_config(self):
        self.assertGreaterEqual(len(self.selector.hospitals), 3)
        h_ids = [h['id'] for h in self.selector.hospitals]
        self.assertIn("H1", h_ids)
        self.assertIn("H2", h_ids)
        self.assertIn("H3", h_ids)

    def test_evaluation_with_mock_traci(self):
        import traci
        # Mock traci.simulation.findRoute
        def mock_find_route(from_edge, to_edge, vType="ambulance"):
            mock_route = MagicMock()
            mock_route.edges = [from_edge, "intermediate", to_edge]
            # Give H1 100s, H2 180s, H3 130s
            if "hosp_to" in from_edge or "to_hosp" in to_edge:
                mock_route.travelTime = 50.0  # H1 total 100s
            elif "hosp_2" in from_edge or "hosp_2" in to_edge:
                mock_route.travelTime = 90.0  # H2 total 180s
            else:
                mock_route.travelTime = 65.0  # H3 total 130s
            return mock_route

        original_find_route = getattr(traci.simulation, "findRoute", None)
        traci.simulation.findRoute = mock_find_route

        try:
            res = self.selector.evaluate_hospitals("ax_3_6_n_to_j_4_6", emergency_severity="CRITICAL_TRAUMA")
            self.assertEqual(res['selected_hospital_id'], "H1")
            self.assertIn("Fastest total travel time", res['selection_reason'])
            self.assertEqual(len(res['candidates']), 3)
            print("\n[TEST SUCCESS] Hospital Selector evaluated correctly:")
            print(f"  Selected: {res['selected_hospital_name']} ({res['selected_hospital_id']})")
            print(f"  Reason: {res['selection_reason']}")
        finally:
            if original_find_route:
                traci.simulation.findRoute = original_find_route

if __name__ == "__main__":
    unittest.main()
