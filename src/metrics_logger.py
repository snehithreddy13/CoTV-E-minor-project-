"""
Performance Metrics Logger for SUMO Simulation
Synchronized from cotv 12 standards.
Tracks ambulance response times, traffic density, and dispatch efficiency.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import traci

class PerformanceMetrics:
    """Tracks and logs structured simulation performance metrics."""
    
    def __init__(self, log_dir: str = 'logs') -> None:
        """Initialize the logger into the specified directory footprint."""
        # Log directory relative to workspace root
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Metrics storage: Dictionary of lists to handle multiple trips per vehicle
        self.ambulance_data: Dict[str, List[Dict[str, Any]]] = {}
        self.traffic_data: List[Dict[str, float]] = []
        self.all_response_times: List[float] = []
        
        self.system_metrics: Dict[str, Any] = {
            'start_time': time.time(),
            'total_vehicles': 0,
            'total_ambulances': 0,
            'simulation_steps': 0
        }
        
    def track_ambulance_dispatch(
        self,
        amb_id: str,
        target: str,
        dispatch_time: int,
        ideal_time: Optional[float] = None,
        emergency_id: str = "",
        selected_hospital_id: str = "",
        selected_hospital_name: str = "",
        selection_reason: str = "",
        candidates_summary: str = ""
    ) -> None:
        """Log a new dispatch event with hospital selection metadata."""
        if amb_id not in self.ambulance_data:
            self.ambulance_data[amb_id] = []
            
        self.ambulance_data[amb_id].append({
            'emergency_id': emergency_id,
            'dispatch_time': dispatch_time,
            'target': target,
            'ideal_time': ideal_time,
            'selected_hospital_id': selected_hospital_id,
            'selected_hospital_name': selected_hospital_name,
            'selection_reason': selection_reason,
            'candidates_summary': candidates_summary,
            'rerouting_occurred': False,
            'status': 'dispatched',
            'arrival_time': None,
            'response_time': None,
            'return_time': None,
            'latencies': [],
            'avg_inference_latency_ms': 0.0
        })

    def track_rerouting(self, amb_id: str) -> None:
        """Flag that a dynamic junction reroute occurred during transit."""
        if amb_id in self.ambulance_data and self.ambulance_data[amb_id]:
            self.ambulance_data[amb_id][-1]['rerouting_occurred'] = True
        
    def track_inference_latency(self, amb_id: str, latency_ms: float) -> None:
        """Log action decision execution latency in milliseconds for active mission of ambulance."""
        if amb_id in self.ambulance_data and self.ambulance_data[amb_id]:
            last_trip = self.ambulance_data[amb_id][-1]
            if 'latencies' not in last_trip:
                last_trip['latencies'] = []
            last_trip['latencies'].append(latency_ms)
            last_trip['avg_inference_latency_ms'] = sum(last_trip['latencies']) / len(last_trip['latencies'])

    def track_ambulance_arrival(self, amb_id: str, arrival_time: int) -> None:
        """Log arrival for the current active mission of the ambulance."""
        if amb_id in self.ambulance_data and self.ambulance_data[amb_id]:
            # Always update the most recent trip
            last_trip = self.ambulance_data[amb_id][-1]
            if last_trip['status'] == 'dispatched':
                last_trip['status'] = 'arrived'
                last_trip['arrival_time'] = arrival_time
                response_time = arrival_time - last_trip['dispatch_time']
                last_trip['response_time'] = response_time
                self.all_response_times.append(response_time)
            
    def track_ambulance_return(self, amb_id: str, return_time: int) -> None:
        """Log return for the current active mission of the ambulance."""
        if amb_id in self.ambulance_data and self.ambulance_data[amb_id]:
            last_trip = self.ambulance_data[amb_id][-1]
            last_trip['status'] = 'returned'
            total_time = return_time - last_trip['dispatch_time']
            last_trip['return_time'] = total_time
    
    def track_traffic_state(self, step: int) -> None:
        """Collect global traffic flow density and speed metrics for the designated simulation step."""
        try:
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                return
                
            waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
            
            self.traffic_data.append({
                'step': step,
                'num_vehicles': float(len(vehicles)),
                'avg_waiting_time': sum(waiting_times) / len(waiting_times),
                'avg_speed': sum(speeds) / len(speeds)
            })
        except traci.TraCIException:
            # TraCI error during vehicle iteration gracefully skipped
            pass
    
    def save_metrics(self, filename: str = 'simulation_metrics') -> str:
        """Save formatted metrics footprint to structured JSON block under logging dir."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.log_dir, f'{filename}_{timestamp}.json')
        
        # Calculate summary statistics
        response_times = self.all_response_times
        total_started = sum(len(trips) for trips in self.ambulance_data.values())
        
        # Calculate aggregate ideal time and inference latency
        it_list = [trip.get('ideal_time') for trips in self.ambulance_data.values() for trip in trips if trip.get('ideal_time') is not None]
        avg_it = sum(it_list) / len(it_list) if it_list else 0

        all_latencies = [l for trips in self.ambulance_data.values() for trip in trips for l in trip.get('latencies', [])]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        summary = {
            'simulation_duration_real': time.time() - self.system_metrics['start_time'],
            'total_ambulances': len(self.ambulance_data),
            'total_missions_started': total_started,
            'total_missions_arrived': len(response_times),
            'avg_ambulance_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'avg_ideal_response_time': avg_it,
            'avg_inference_latency_ms': avg_latency,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
        }
        
        full_data = {
            'summary': summary,
            'ambulances': self.ambulance_data,
            'traffic_timeline': self.traffic_data[-200:], # Keep last 200 samples
        }
        
        with open(filepath, 'w') as f:
            json.dump(full_data, f, indent=2)
        
        print(f"\n[METRICS] Performance report saved to: {filepath}")

        # Also write CSV log to database/hospital_dispatch_history.csv
        try:
            import csv
            db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../database")
            os.makedirs(db_dir, exist_ok=True)
            csv_path = os.path.join(db_dir, "hospital_dispatch_history.csv")
            file_exists = os.path.isfile(csv_path)

            fieldnames = [
                'timestamp', 'emergency_id', 'ambulance_id', 'emergency_location',
                'selected_hospital_id', 'selected_hospital_name', 'est_travel_time',
                'selection_reason', 'dispatch_step', 'arrival_step',
                'response_time_s', 'rerouting_occurred'
            ]

            with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                for amb_id, trips in self.ambulance_data.items():
                    for trip in trips:
                        writer.writerow({
                            'timestamp': timestamp,
                            'emergency_id': trip.get('emergency_id', ''),
                            'ambulance_id': amb_id,
                            'emergency_location': trip.get('target', ''),
                            'selected_hospital_id': trip.get('selected_hospital_id', ''),
                            'selected_hospital_name': trip.get('selected_hospital_name', ''),
                            'est_travel_time': trip.get('ideal_time', ''),
                            'selection_reason': trip.get('selection_reason', ''),
                            'dispatch_step': trip.get('dispatch_time', ''),
                            'arrival_step': trip.get('arrival_time', ''),
                            'response_time_s': trip.get('response_time', ''),
                            'rerouting_occurred': trip.get('rerouting_occurred', False)
                        })

            print(f"[DATABASE] Hospital dispatch history logged to: {csv_path}")
        except Exception as e:
            print(f"[WARNING] Could not save hospital dispatch history CSV: {e}")

        return filepath
