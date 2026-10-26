import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def build_6x6_original():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nod_filepath = os.path.join(script_dir, "network.nod.xml")
    edg_filepath = os.path.join(script_dir, "network.edg.xml")

    # 1. Nodes
    nodes_root = ET.Element("nodes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"
    })

    # Main grid intersections j_0_0 to j_5_5
    for r in range(6):
        for c in range(6):
            ET.SubElement(nodes_root, "node", {
                "id": f"j_{c}_{r}",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "traffic_light"
            })

    # Aux nodes horizontal (ax_{c}_{r}_n, ax_{c}_{r}_s)
    for r in range(6):
        for c in range(5):
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_n",
                "x": f"{c * 600.0 + 300.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "priority"
            })
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_s",
                "x": f"{c * 600.0 + 300.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "priority"
            })

    # Aux nodes vertical (ax_{c}_{r}_e, ax_{c}_{r}_w)
    for c in range(6):
        for r in range(5):
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_e",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0 + 300.0:.1f}",
                "type": "priority"
            })
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_w",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0 + 300.0:.1f}",
                "type": "priority"
            })

    # Hospital center & junction nodes
    ET.SubElement(nodes_root, "node", {"id": "hosp", "x": "1500.0", "y": "1500.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_n", "x": "1500.0", "y": "1600.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_s", "x": "1500.0", "y": "1400.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_e", "x": "1600.0", "y": "1500.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_w", "x": "1400.0", "y": "1500.0", "type": "priority"})

    with open(nod_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(nodes_root))

    # 2. Edges
    edges_root = ET.Element("edges", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"
    })

    edge_kwargs = {"numLanes": "2", "speed": "13.89", "priority": "3", "width": "16.0"}

    # Horizontal edges connecting j_{c}_{r} <-> ax_{c}_{r}_n <-> j_{c+1}_{r}
    for r in range(6):
        for c in range(5):
            # Special case for hospital zone in row 2 and row 3
            if r == 2 and c == 2:
                # Between j_2_2 and j_3_2 via j_hosp_n
                ET.SubElement(edges_root, "edge", {"id": "j_2_2_to_j_hosp_n", "from": "j_2_2", "to": "j_hosp_n", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_j_2_2", "from": "j_hosp_n", "to": "j_2_2", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_ax_2_1_s", "from": "j_hosp_n", "to": "ax_2_1_s", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_1_s_to_j_hosp_n", "from": "ax_2_1_s", "to": "j_hosp_n", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_1_s_to_j_3_2", "from": "ax_2_1_s", "to": "j_3_2", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_3_2_to_ax_2_1_s", "from": "j_3_2", "to": "ax_2_1_s", **edge_kwargs})
            elif r == 3 and c == 2:
                # Between j_2_3 and j_3_3 via j_hosp_s
                ET.SubElement(edges_root, "edge", {"id": "j_2_3_to_ax_2_3_n", "from": "j_2_3", "to": "ax_2_3_n", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_3_n_to_j_2_3", "from": "ax_2_3_n", "to": "j_2_3", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_3_n_to_j_hosp_s", "from": "ax_2_3_n", "to": "j_hosp_s", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_ax_2_3_n", "from": "j_hosp_s", "to": "ax_2_3_n", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_j_3_3", "from": "j_hosp_s", "to": "j_3_3", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_3_3_to_j_hosp_s", "from": "j_3_3", "to": "j_hosp_s", **edge_kwargs})
            else:
                n_left = f"j_{c}_{r}"
                n_mid = f"ax_{c}_{r}_n"
                n_right = f"j_{c+1}_{r}"

                ET.SubElement(edges_root, "edge", {"id": f"{n_left}_to_{n_mid}", "from": n_left, "to": n_mid, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_left}", "from": n_mid, "to": n_left, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_right}", "from": n_mid, "to": n_right, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_right}_to_{n_mid}", "from": n_right, "to": n_mid, **edge_kwargs})

                if r > 0 and r < 5:
                    s_mid = f"ax_{c}_{r-1}_s"
                    ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{s_mid}", "from": n_mid, "to": s_mid, **edge_kwargs})
                    ET.SubElement(edges_root, "edge", {"id": f"{s_mid}_to_{n_mid}", "from": s_mid, "to": n_mid, **edge_kwargs})

    # Vertical edges connecting j_{c}_{r} <-> ax_{c}_{r}_w <-> j_{c}_{r+1}
    for c in range(6):
        for r in range(5):
            if c == 2 and r == 2:
                # Between j_2_2 and j_2_3 via j_hosp_w
                ET.SubElement(edges_root, "edge", {"id": "j_2_2_to_ax_1_2_e", "from": "j_2_2", "to": "ax_1_2_e", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_2_e_to_j_2_2", "from": "ax_1_2_e", "to": "j_2_2", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_2_e_to_j_hosp_w", "from": "ax_1_2_e", "to": "j_hosp_w", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_ax_1_2_e", "from": "j_hosp_w", "to": "ax_1_2_e", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_j_2_3", "from": "j_hosp_w", "to": "j_2_3", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_2_3_to_j_hosp_w", "from": "j_2_3", "to": "j_hosp_w", **edge_kwargs})
            elif c == 3 and r == 2:
                # Between j_3_2 and j_3_3 via j_hosp_e
                ET.SubElement(edges_root, "edge", {"id": "j_3_2_to_j_hosp_e", "from": "j_3_2", "to": "j_hosp_e", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_j_3_2", "from": "j_hosp_e", "to": "j_3_2", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_ax_3_2_w", "from": "j_hosp_e", "to": "ax_3_2_w", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_3_2_w_to_j_hosp_e", "from": "ax_3_2_w", "to": "j_hosp_e", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "ax_3_2_w_to_j_3_3", "from": "ax_3_2_w", "to": "j_3_3", **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": "j_3_3_to_ax_3_2_w", "from": "j_3_3", "to": "ax_3_2_w", **edge_kwargs})
            else:
                n_bot = f"j_{c}_{r}"
                n_mid = f"ax_{c}_{r}_w"
                n_top = f"j_{c}_{r+1}"

                ET.SubElement(edges_root, "edge", {"id": f"{n_bot}_to_{n_mid}", "from": n_bot, "to": n_mid, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_bot}", "from": n_mid, "to": n_bot, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_top}", "from": n_mid, "to": n_top, **edge_kwargs})
                ET.SubElement(edges_root, "edge", {"id": f"{n_top}_to_{n_mid}", "from": n_top, "to": n_mid, **edge_kwargs})

                if c > 0 and c < 5:
                    e_mid = f"ax_{c-1}_{r}_e"
                    ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{e_mid}", "from": n_mid, "to": e_mid, **edge_kwargs})
                    ET.SubElement(edges_root, "edge", {"id": f"{e_mid}_to_{n_mid}", "from": e_mid, "to": n_mid, **edge_kwargs})

    # Hospital center connections
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_n", "from": "hosp", "to": "j_hosp_n", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_hosp", "from": "j_hosp_n", "to": "hosp", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_s", "from": "hosp", "to": "j_hosp_s", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_hosp", "from": "j_hosp_s", "to": "hosp", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_e", "from": "hosp", "to": "j_hosp_e", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_hosp", "from": "j_hosp_e", "to": "hosp", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_w", "from": "hosp", "to": "j_hosp_w", **edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_hosp", "from": "j_hosp_w", "to": "hosp", **edge_kwargs})

    with open(edg_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(edges_root))

    print("[SUCCESS] Restored 6x6 original nod and edg XML files.")

if __name__ == "__main__":
    build_6x6_original()
