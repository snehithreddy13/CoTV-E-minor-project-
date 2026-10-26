"""
Perception Module: Simulates Drone-based Video Monitoring and YOLO Object Detection.
Synchronized from cotv 12.
"""

import numpy as np
import traci

from typing import Any, Dict, List

class PerceptionSystem:
    """
    Simulates the pipeline: Drone Video -> YOLO Detection -> Traffic Feature Extraction.
    """
    
    def __init__(self, traci_connection: Any) -> None:
        """
        Initializes the perception system with an active TraCI connection footprint.
        """
        self.traci = traci_connection
        
    def get_drone_observation(self, junction_id: str) -> Dict[str, float]:
        """
        Simulates extracting features for a specific junction as seen by a drone.
        """
        try:
            lanes = self.traci.trafficlight.getControlledLanes(junction_id)
            total_vehicles = 0
            speeds = []
            max_queue = 0
            ambulance_detected = False
            
            for lane in lanes:
                veh_ids = self.traci.lane.getLastStepVehicleIDs(lane)
                total_vehicles += len(veh_ids)
                for v in veh_ids:
                    speeds.append(self.traci.vehicle.getSpeed(v))
                    if 'ambulance' in v.lower():
                        ambulance_detected = True
                
                queue = self.traci.lane.getLastStepHaltingNumber(lane)
                if queue > max_queue:
                    max_queue = queue
            
            avg_speed = np.mean(speeds) if speeds else 0.0
            
            return {
                'vehicle_count': min(total_vehicles / 20.0, 1.0),
                'avg_speed': min(avg_speed / 20.0, 1.0),
                'max_queue': min(max_queue / 20.0, 1.0),
                'ambulance_present': 1.0 if ambulance_detected else 0.0
            }
        except traci.TraCIException as e:
            # Safely trap specific traci disconnects instead of broad exceptions
            return {'vehicle_count': 0.0, 'avg_speed': 0.0, 'max_queue': 0.0, 'ambulance_present': 0.0}

    def detect_emergency_vehicles(self) -> List[Dict[str, Any]]:
        """
        Scans all visible areas for ambulances using a simulated YOLO architecture logic layer.
        
        Returns:
            A list of dictionary objects describing detected active emergencies.
        """
        emergency_list: List[Dict[str, Any]] = []
        try:
            vehicle_ids = self.traci.vehicle.getIDList()
            for v_id in vehicle_ids:
                if 'ambulance' in v_id.lower():
                    lane_id = self.traci.vehicle.getLaneID(v_id)
                    next_tls = self.traci.vehicle.getNextTLS(v_id)
                    if next_tls:
                        tls_id = next_tls[0][0]
                        dist = next_tls[0][2]
                        emergency_list.append({
                            'id': v_id,
                            'lane': lane_id,
                            'junction': tls_id,
                            'distance': dist
                        })
        except traci.TraCIException:
            # TraCI iteration edge case safeguard for despawning vehicles
            pass
            
        return emergency_list
