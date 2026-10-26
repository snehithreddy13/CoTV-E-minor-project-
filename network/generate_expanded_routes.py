import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import sumolib

def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def generate_expanded_routes_and_cfg():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(script_dir, "network.net.xml")
    rou_filepath = os.path.join(script_dir, "routes.rou.xml")
    cfg_filepath = os.path.join(script_dir, "simulation.sumocfg")

    if not os.path.exists(net_path):
        print(f"[ERROR] Network file not found: {net_path}")
        sys.exit(1)

    print(f"[INFO] Loading network for route generation: {net_path}")
    net = sumolib.net.readNet(net_path)

    routes_root = ET.Element("routes")

    # We will generate horizontal and vertical flow corridors for an 8x8 grid
    # Horizontal: row 0 to 7
    # Vertical: col 0 to 7

    grid_cols = 8
    grid_rows = 8

    route_count = 0
    flow_count = 0

    vehicle_types = ["sedan", "motorcycle", "lorry"]
    type_probabilities = {"sedan": "0.015", "motorcycle": "0.005", "lorry": "0.002"}

    # Horizontal routes (Eastbound & Westbound)
    for r in range(grid_rows):
        # Eastbound: start from edge near j_0_{r} to edge near j_{grid_cols-1}_{r}
        start_edge = net.getEdge(f"j_0_{r}_to_ax_0_{r}_n")
        end_edge = net.getEdge(f"ax_{grid_cols-2}_{r}_n_to_j_{grid_cols-1}_{r}")
        path_east = net.getShortestPath(start_edge, end_edge, vClass="passenger")
        if path_east[0]:
            edges_str = " ".join([e.getID() for e in path_east[0]])
            route_id = f"h_east_{r}"
            ET.SubElement(routes_root, "route", {"id": route_id, "edges": edges_str})
            route_count += 1

            for vtype in vehicle_types:
                flow_id = f"flow_h_east_{r}_{vtype}"
                ET.SubElement(routes_root, "flow", {
                    "id": flow_id,
                    "type": vtype,
                    "route": route_id,
                    "begin": "0",
                    "end": "1000000",
                    "probability": type_probabilities[vtype]
                })
                flow_count += 1

        # Westbound: start from edge near j_{grid_cols-1}_{r} to edge near j_0_{r}
        start_edge = net.getEdge(f"j_{grid_cols-1}_{r}_to_ax_{grid_cols-2}_{r}_n")
        end_edge = net.getEdge(f"ax_0_{r}_n_to_j_0_{r}")
        path_west = net.getShortestPath(start_edge, end_edge, vClass="passenger")
        if path_west[0]:
            edges_str = " ".join([e.getID() for e in path_west[0]])
            route_id = f"h_west_{r}"
            ET.SubElement(routes_root, "route", {"id": route_id, "edges": edges_str})
            route_count += 1

            for vtype in vehicle_types:
                flow_id = f"flow_h_west_{r}_{vtype}"
                ET.SubElement(routes_root, "flow", {
                    "id": flow_id,
                    "type": vtype,
                    "route": route_id,
                    "begin": "0",
                    "end": "1000000",
                    "probability": type_probabilities[vtype]
                })
                flow_count += 1

    # Vertical routes (Northbound & Southbound)
    for c in range(grid_cols):
        # Northbound: start from j_{c}_0 to j_{c}_{grid_rows-1}
        start_edge = net.getEdge(f"j_{c}_0_to_ax_{c}_0_w")
        end_edge = net.getEdge(f"ax_{c}_{grid_rows-2}_w_to_j_{c}_{grid_rows-1}")
        path_north = net.getShortestPath(start_edge, end_edge, vClass="passenger")
        if path_north[0]:
            edges_str = " ".join([e.getID() for e in path_north[0]])
            route_id = f"v_north_{c}"
            ET.SubElement(routes_root, "route", {"id": route_id, "edges": edges_str})
            route_count += 1

            for vtype in vehicle_types:
                flow_id = f"flow_v_north_{c}_{vtype}"
                ET.SubElement(routes_root, "flow", {
                    "id": flow_id,
                    "type": vtype,
                    "route": route_id,
                    "begin": "0",
                    "end": "1000000",
                    "probability": type_probabilities[vtype]
                })
                flow_count += 1

        # Southbound: start from j_{c}_{grid_rows-1} to j_{c}_0
        start_edge = net.getEdge(f"j_{c}_{grid_rows-1}_to_ax_{c}_{grid_rows-2}_w")
        end_edge = net.getEdge(f"ax_{c}_0_w_to_j_{c}_0")
        path_south = net.getShortestPath(start_edge, end_edge, vClass="passenger")
        if path_south[0]:
            edges_str = " ".join([e.getID() for e in path_south[0]])
            route_id = f"v_south_{c}"
            ET.SubElement(routes_root, "route", {"id": route_id, "edges": edges_str})
            route_count += 1

            for vtype in vehicle_types:
                flow_id = f"flow_v_south_{c}_{vtype}"
                ET.SubElement(routes_root, "flow", {
                    "id": flow_id,
                    "type": vtype,
                    "route": route_id,
                    "begin": "0",
                    "end": "1000000",
                    "probability": type_probabilities[vtype]
                })
                flow_count += 1

    with open(rou_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(routes_root))

    print(f"[SUCCESS] Generated {route_count} routes and {flow_count} background traffic flows in {rou_filepath}")

    # Generate simulation.sumocfg
    cfg_content = """<?xml version="1.0" encoding="UTF-8"?>
<!-- CoTV-E SUMO Simulation Configuration (8x8 Urban Grid + Multi-Hospital) -->
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="buildings.add.xml,vehicle_types.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="1000000"/>
        <step-length value="1.0"/>
    </time>

    <processing>
        <time-to-teleport value="60"/>
        <time-to-teleport.highways value="45"/>
        <collision.action value="warn"/>
        <routing-algorithm value="dijkstra"/>
    </processing>

    <routing>
        <device.rerouting.probability value="0.7"/>
        <device.rerouting.period value="60"/>
        <device.rerouting.adaptation-steps value="180"/>
    </routing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="gui-settings.xml"/>
    </gui_only>

</configuration>
"""
    with open(cfg_filepath, "w", encoding="utf-8") as f:
        f.write(cfg_content)

    print(f"[SUCCESS] Generated expanded simulation configuration: {cfg_filepath}")

if __name__ == "__main__":
    generate_expanded_routes_and_cfg()
