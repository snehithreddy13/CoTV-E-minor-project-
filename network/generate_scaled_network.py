import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def generate_network_files(grid_cols: int = 4, grid_rows: int = 4, spacing: float = 500.0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nod_filepath = os.path.join(script_dir, "scaled_network.nod.xml")
    edg_filepath = os.path.join(script_dir, "scaled_network.edg.xml")

    # -------------------------------------------------------------------------
    # 1. Generate Nodes (network.nod.xml)
    # -------------------------------------------------------------------------
    nodes_root = ET.Element("nodes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"
    })

    # Grid Intersection Nodes
    for row in range(grid_rows):
        for col in range(grid_cols):
            node_id = f"j_{col}_{row}"
            x_pos = f"{col * spacing:.1f}"
            y_pos = f"{row * spacing:.1f}"
            ET.SubElement(nodes_root, "node", {
                "id": node_id,
                "x": x_pos,
                "y": y_pos,
                "type": "traffic_light"
            })

    # Hospital Nodes (hosp_1 and hosp_2 at different ends of grid)
    # hosp_1 placed near bottom-left intersection (j_0_0)
    ET.SubElement(nodes_root, "node", {
        "id": "hosp_1",
        "x": f"{-spacing / 2:.1f}",
        "y": "0.0",
        "type": "priority"
    })

    # hosp_2 placed near top-right intersection (j_{cols-1}_{rows-1})
    ET.SubElement(nodes_root, "node", {
        "id": "hosp_2",
        "x": f"{(grid_cols - 1) * spacing + spacing / 2:.1f}",
        "y": f"{(grid_rows - 1) * spacing:.1f}",
        "type": "priority"
    })

    # Write network.nod.xml
    with open(nod_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(nodes_root))
    print(f"[SUCCESS] Generated: {nod_filepath}")

    # -------------------------------------------------------------------------
    # 2. Generate Edges (network.edg.xml)
    # -------------------------------------------------------------------------
    edges_root = ET.Element("edges", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"
    })

    edge_kwargs = {
        "numLanes": "2",
        "speed": "13.89",
        "priority": "3",
        "width": "16.0"
    }

    # Horizontal two-way grid edges
    for row in range(grid_rows):
        for col in range(grid_cols - 1):
            n1 = f"j_{col}_{row}"
            n2 = f"j_{col + 1}_{row}"
            
            # Forward edge
            ET.SubElement(edges_root, "edge", {
                "id": f"{n1}_to_{n2}",
                "from": n1,
                "to": n2,
                **edge_kwargs
            })
            # Backward edge
            ET.SubElement(edges_root, "edge", {
                "id": f"{n2}_to_{n1}",
                "from": n2,
                "to": n1,
                **edge_kwargs
            })

    # Vertical two-way grid edges
    for col in range(grid_cols):
        for row in range(grid_rows - 1):
            n1 = f"j_{col}_{row}"
            n2 = f"j_{col}_{row + 1}"

            # Forward edge
            ET.SubElement(edges_root, "edge", {
                "id": f"{n1}_to_{n2}",
                "from": n1,
                "to": n2,
                **edge_kwargs
            })
            # Backward edge
            ET.SubElement(edges_root, "edge", {
                "id": f"{n2}_to_{n1}",
                "from": n2,
                "to": n1,
                **edge_kwargs
            })

    # Hospital 1 Connectivity (connected to j_0_0)
    ET.SubElement(edges_root, "edge", {
        "id": "hosp_1_to_j_0_0",
        "from": "hosp_1",
        "to": "j_0_0",
        **edge_kwargs
    })
    ET.SubElement(edges_root, "edge", {
        "id": "j_0_0_to_hosp_1",
        "from": "j_0_0",
        "to": "hosp_1",
        **edge_kwargs
    })

    # Hospital 2 Connectivity (connected to j_{grid_cols-1}_{grid_rows-1})
    top_right_node = f"j_{grid_cols - 1}_{grid_rows - 1}"
    ET.SubElement(edges_root, "edge", {
        "id": f"hosp_2_to_{top_right_node}",
        "from": "hosp_2",
        "to": top_right_node,
        **edge_kwargs
    })
    ET.SubElement(edges_root, "edge", {
        "id": f"{top_right_node}_to_hosp_2",
        "from": top_right_node,
        "to": "hosp_2",
        **edge_kwargs
    })

    # Write network.edg.xml
    with open(edg_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(edges_root))
    print(f"[SUCCESS] Generated: {edg_filepath}")

    print("\n" + "=" * 65)
    print("NETCONVERT COMPILATION COMMAND:")
    print("=" * 65)
    print("netconvert --node-files=network/scaled_network.nod.xml --edge-files=network/scaled_network.edg.xml --output-file=network/scaled_network.net.xml --no-turnarounds=false")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    generate_network_files(grid_cols=4, grid_rows=4, spacing=500.0)
