import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess
import sys

def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def build_expanded_network(grid_cols: int = 8, grid_rows: int = 8):
    """
    Builds an expanded 8x8 urban network (6x6 original + 2 additional city blocks)
    with three hospital nodes placed in distinct city sectors:
    - Hospital 1 (hosp / hosp_1): Central hub at (1500, 1500)
    - Hospital 2 (hosp_2): Northeast hub at (3300, 3300)
    - Hospital 3 (hosp_3): Northwest hub at (900, 3300)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nod_filepath = os.path.join(script_dir, "network.nod.xml")
    edg_filepath = os.path.join(script_dir, "network.edg.xml")
    net_filepath = os.path.join(script_dir, "network.net.xml")

    # 1. Nodes XML
    nodes_root = ET.Element("nodes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"
    })

    # Main grid intersections j_0_0 to j_{cols-1}_{rows-1}
    for r in range(grid_rows):
        for c in range(grid_cols):
            ET.SubElement(nodes_root, "node", {
                "id": f"j_{c}_{r}",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "traffic_light"
            })

    # Aux nodes horizontal (ax_{c}_{r}_n, ax_{c}_{r}_s)
    for r in range(grid_rows):
        for c in range(grid_cols - 1):
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_n",
                "x": f"{c * 600.0 + 300.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "traffic_light"
            })
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_s",
                "x": f"{c * 600.0 + 300.0:.1f}",
                "y": f"{r * 600.0:.1f}",
                "type": "traffic_light"
            })

    # Aux nodes vertical (ax_{c}_{r}_e, ax_{c}_{r}_w)
    for c in range(grid_cols):
        for r in range(grid_rows - 1):
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_e",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0 + 300.0:.1f}",
                "type": "traffic_light"
            })
            ET.SubElement(nodes_root, "node", {
                "id": f"ax_{c}_{r}_w",
                "x": f"{c * 600.0:.1f}",
                "y": f"{r * 600.0 + 300.0:.1f}",
                "type": "traffic_light"
            })

    # Hospital 1 center & junction nodes (Original central hub at 1500, 1500)
    ET.SubElement(nodes_root, "node", {"id": "hosp", "x": "1500.0", "y": "1500.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_n", "x": "1500.0", "y": "1600.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_s", "x": "1500.0", "y": "1400.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_e", "x": "1600.0", "y": "1500.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp_w", "x": "1400.0", "y": "1500.0", "type": "traffic_light"})

    # Hospital 2 center & junction nodes (Northeast hub at 3300, 3300)
    ET.SubElement(nodes_root, "node", {"id": "hosp_2", "x": "3300.0", "y": "3300.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp2_n", "x": "3300.0", "y": "3400.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp2_s", "x": "3300.0", "y": "3200.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp2_e", "x": "3400.0", "y": "3300.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp2_w", "x": "3200.0", "y": "3300.0", "type": "traffic_light"})

    # Hospital 3 center & junction nodes (Northwest hub at 900, 3300)
    ET.SubElement(nodes_root, "node", {"id": "hosp_3", "x": "900.0", "y": "3300.0", "type": "priority"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp3_n", "x": "900.0", "y": "3400.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp3_s", "x": "900.0", "y": "3200.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp3_e", "x": "1000.0", "y": "3300.0", "type": "traffic_light"})
    ET.SubElement(nodes_root, "node", {"id": "j_hosp3_w", "x": "800.0", "y": "3300.0", "type": "traffic_light"})

    with open(nod_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(nodes_root))

    # 2. Edges XML
    edges_root = ET.Element("edges", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"
    })

    edge_kwargs = {"numLanes": "2", "speed": "13.89", "priority": "3", "width": "16.0"}
    hosp_edge_kwargs = {**edge_kwargs, "disallow": "passenger delivery truck bus motorcycle trailer moped"}

    def get_kw(n1: str, n2: str) -> dict:
        if "hosp" in n1 or "hosp" in n2:
            return hosp_edge_kwargs
        return edge_kwargs

    # Horizontal edges connecting j_{c}_{r} <-> ax_{c}_{r}_n/s <-> j_{c+1}_{r}
    for r in range(grid_rows):
        for c in range(grid_cols - 1):
            # Special case for Hospital 1 zone in row 2 and row 3, col 2
            if r == 2 and c == 2:
                ET.SubElement(edges_root, "edge", {"id": "j_2_2_to_j_hosp_n", "from": "j_2_2", "to": "j_hosp_n", **get_kw("j_2_2", "j_hosp_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_j_2_2", "from": "j_hosp_n", "to": "j_2_2", **get_kw("j_hosp_n", "j_2_2")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_ax_2_1_s", "from": "j_hosp_n", "to": "ax_2_1_s", **get_kw("j_hosp_n", "ax_2_1_s")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_1_s_to_j_hosp_n", "from": "ax_2_1_s", "to": "j_hosp_n", **get_kw("ax_2_1_s", "j_hosp_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_1_s_to_j_3_2", "from": "ax_2_1_s", "to": "j_3_2", **get_kw("ax_2_1_s", "j_3_2")})
                ET.SubElement(edges_root, "edge", {"id": "j_3_2_to_ax_2_1_s", "from": "j_3_2", "to": "ax_2_1_s", **get_kw("j_3_2", "ax_2_1_s")})
            elif r == 3 and c == 2:
                ET.SubElement(edges_root, "edge", {"id": "j_2_3_to_ax_2_3_n", "from": "j_2_3", "to": "ax_2_3_n", **get_kw("j_2_3", "ax_2_3_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_3_n_to_j_2_3", "from": "ax_2_3_n", "to": "j_2_3", **get_kw("ax_2_3_n", "j_2_3")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_3_n_to_j_hosp_s", "from": "ax_2_3_n", "to": "j_hosp_s", **get_kw("ax_2_3_n", "j_hosp_s")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_ax_2_3_n", "from": "j_hosp_s", "to": "ax_2_3_n", **get_kw("j_hosp_s", "ax_2_3_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_j_3_3", "from": "j_hosp_s", "to": "j_3_3", **get_kw("j_hosp_s", "j_3_3")})
                ET.SubElement(edges_root, "edge", {"id": "j_3_3_to_j_hosp_s", "from": "j_3_3", "to": "j_hosp_s", **get_kw("j_3_3", "j_hosp_s")})
            # Special case for Hospital 2 zone in row 5 and row 6, col 5
            elif r == 5 and c == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_5_5_to_j_hosp2_n", "from": "j_5_5", "to": "j_hosp2_n", **get_kw("j_5_5", "j_hosp2_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_n_to_j_5_5", "from": "j_hosp2_n", "to": "j_5_5", **get_kw("j_hosp2_n", "j_5_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_n_to_ax_5_4_s", "from": "j_hosp2_n", "to": "ax_5_4_s", **get_kw("j_hosp2_n", "ax_5_4_s")})
                ET.SubElement(edges_root, "edge", {"id": "ax_5_4_s_to_j_hosp2_n", "from": "ax_5_4_s", "to": "j_hosp2_n", **get_kw("ax_5_4_s", "j_hosp2_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_5_4_s_to_j_6_5", "from": "ax_5_4_s", "to": "j_6_5", **get_kw("ax_5_4_s", "j_6_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_6_5_to_ax_5_4_s", "from": "j_6_5", "to": "ax_5_4_s", **get_kw("j_6_5", "ax_5_4_s")})
            elif r == 6 and c == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_5_6_to_ax_5_6_n", "from": "j_5_6", "to": "ax_5_6_n", **get_kw("j_5_6", "ax_5_6_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_5_6_n_to_j_5_6", "from": "ax_5_6_n", "to": "j_5_6", **get_kw("ax_5_6_n", "j_5_6")})
                ET.SubElement(edges_root, "edge", {"id": "ax_5_6_n_to_j_hosp2_s", "from": "ax_5_6_n", "to": "j_hosp2_s", **get_kw("ax_5_6_n", "j_hosp2_s")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_s_to_ax_5_6_n", "from": "j_hosp2_s", "to": "ax_5_6_n", **get_kw("j_hosp2_s", "ax_5_6_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_s_to_j_6_6", "from": "j_hosp2_s", "to": "j_6_6", **get_kw("j_hosp2_s", "j_6_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_6_6_to_j_hosp2_s", "from": "j_6_6", "to": "j_hosp2_s", **get_kw("j_6_6", "j_hosp2_s")})
            # Special case for Hospital 3 zone in row 5 and row 6, col 1
            elif r == 5 and c == 1:
                ET.SubElement(edges_root, "edge", {"id": "j_1_5_to_j_hosp3_n", "from": "j_1_5", "to": "j_hosp3_n", **get_kw("j_1_5", "j_hosp3_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_n_to_j_1_5", "from": "j_hosp3_n", "to": "j_1_5", **get_kw("j_hosp3_n", "j_1_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_n_to_ax_1_4_s", "from": "j_hosp3_n", "to": "ax_1_4_s", **get_kw("j_hosp3_n", "ax_1_4_s")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_4_s_to_j_hosp3_n", "from": "ax_1_4_s", "to": "j_hosp3_n", **get_kw("ax_1_4_s", "j_hosp3_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_4_s_to_j_2_5", "from": "ax_1_4_s", "to": "j_2_5", **get_kw("ax_1_4_s", "j_2_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_2_5_to_ax_1_4_s", "from": "j_2_5", "to": "ax_1_4_s", **get_kw("j_2_5", "ax_1_4_s")})
            elif r == 6 and c == 1:
                ET.SubElement(edges_root, "edge", {"id": "j_1_6_to_ax_1_6_n", "from": "j_1_6", "to": "ax_1_6_n", **get_kw("j_1_6", "ax_1_6_n")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_6_n_to_j_1_6", "from": "ax_1_6_n", "to": "j_1_6", **get_kw("ax_1_6_n", "j_1_6")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_6_n_to_j_hosp3_s", "from": "ax_1_6_n", "to": "j_hosp3_s", **get_kw("ax_1_6_n", "j_hosp3_s")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_s_to_ax_1_6_n", "from": "j_hosp3_s", "to": "ax_1_6_n", **get_kw("j_hosp3_s", "ax_1_6_n")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_s_to_j_2_6", "from": "j_hosp3_s", "to": "j_2_6", **get_kw("j_hosp3_s", "j_2_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_2_6_to_j_hosp3_s", "from": "j_2_6", "to": "j_hosp3_s", **get_kw("j_2_6", "j_hosp3_s")})
            else:
                n_left = f"j_{c}_{r}"
                n_mid = f"ax_{c}_{r}_n"
                n_right = f"j_{c+1}_{r}"

                ET.SubElement(edges_root, "edge", {"id": f"{n_left}_to_{n_mid}", "from": n_left, "to": n_mid, **get_kw(n_left, n_mid)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_left}", "from": n_mid, "to": n_left, **get_kw(n_mid, n_left)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_right}", "from": n_mid, "to": n_right, **get_kw(n_mid, n_right)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_right}_to_{n_mid}", "from": n_right, "to": n_mid, **get_kw(n_right, n_mid)})

                if r > 0 and r < (grid_rows - 1):
                    s_mid = f"ax_{c}_{r-1}_s"
                    ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{s_mid}", "from": n_mid, "to": s_mid, **get_kw(n_mid, s_mid)})
                    ET.SubElement(edges_root, "edge", {"id": f"{s_mid}_to_{n_mid}", "from": s_mid, "to": n_mid, **get_kw(s_mid, n_mid)})

    # Vertical edges connecting j_{c}_{r} <-> ax_{c}_{r}_w/e <-> j_{c}_{r+1}
    for c in range(grid_cols):
        for r in range(grid_rows - 1):
            if c == 2 and r == 2:
                ET.SubElement(edges_root, "edge", {"id": "j_2_2_to_ax_1_2_e", "from": "j_2_2", "to": "ax_1_2_e", **get_kw("j_2_2", "ax_1_2_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_2_e_to_j_2_2", "from": "ax_1_2_e", "to": "j_2_2", **get_kw("ax_1_2_e", "j_2_2")})
                ET.SubElement(edges_root, "edge", {"id": "ax_1_2_e_to_j_hosp_w", "from": "ax_1_2_e", "to": "j_hosp_w", **get_kw("ax_1_2_e", "j_hosp_w")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_ax_1_2_e", "from": "j_hosp_w", "to": "ax_1_2_e", **get_kw("j_hosp_w", "ax_1_2_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_j_2_3", "from": "j_hosp_w", "to": "j_2_3", **get_kw("j_hosp_w", "j_2_3")})
                ET.SubElement(edges_root, "edge", {"id": "j_2_3_to_j_hosp_w", "from": "j_2_3", "to": "j_hosp_w", **get_kw("j_2_3", "j_hosp_w")})
            elif c == 3 and r == 2:
                ET.SubElement(edges_root, "edge", {"id": "j_3_2_to_j_hosp_e", "from": "j_3_2", "to": "j_hosp_e", **get_kw("j_3_2", "j_hosp_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_j_3_2", "from": "j_hosp_e", "to": "j_3_2", **get_kw("j_hosp_e", "j_3_2")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_ax_3_2_w", "from": "j_hosp_e", "to": "ax_3_2_w", **get_kw("j_hosp_e", "ax_3_2_w")})
                ET.SubElement(edges_root, "edge", {"id": "ax_3_2_w_to_j_hosp_e", "from": "ax_3_2_w", "to": "j_hosp_e", **get_kw("ax_3_2_w", "j_hosp_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_3_2_w_to_j_3_3", "from": "ax_3_2_w", "to": "j_3_3", **get_kw("ax_3_2_w", "j_3_3")})
                ET.SubElement(edges_root, "edge", {"id": "j_3_3_to_ax_3_2_w", "from": "j_3_3", "to": "ax_3_2_w", **get_kw("j_3_3", "ax_3_2_w")})
            elif c == 5 and r == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_5_5_to_ax_4_5_e", "from": "j_5_5", "to": "ax_4_5_e", **get_kw("j_5_5", "ax_4_5_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_4_5_e_to_j_5_5", "from": "ax_4_5_e", "to": "j_5_5", **get_kw("ax_4_5_e", "j_5_5")})
                ET.SubElement(edges_root, "edge", {"id": "ax_4_5_e_to_j_hosp2_w", "from": "ax_4_5_e", "to": "j_hosp2_w", **get_kw("ax_4_5_e", "j_hosp2_w")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_w_to_ax_4_5_e", "from": "j_hosp2_w", "to": "ax_4_5_e", **get_kw("j_hosp2_w", "ax_4_5_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_w_to_j_5_6", "from": "j_hosp2_w", "to": "j_5_6", **get_kw("j_hosp2_w", "j_5_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_5_6_to_j_hosp2_w", "from": "j_5_6", "to": "j_hosp2_w", **get_kw("j_5_6", "j_hosp2_w")})
            elif c == 6 and r == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_6_5_to_j_hosp2_e", "from": "j_6_5", "to": "j_hosp2_e", **get_kw("j_6_5", "j_hosp2_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_e_to_j_6_5", "from": "j_hosp2_e", "to": "j_6_5", **get_kw("j_hosp2_e", "j_6_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp2_e_to_ax_6_5_w", "from": "j_hosp2_e", "to": "ax_6_5_w", **get_kw("j_hosp2_e", "ax_6_5_w")})
                ET.SubElement(edges_root, "edge", {"id": "ax_6_5_w_to_j_hosp2_e", "from": "ax_6_5_w", "to": "j_hosp2_e", **get_kw("ax_6_5_w", "j_hosp2_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_6_5_w_to_j_6_6", "from": "ax_6_5_w", "to": "j_6_6", **get_kw("ax_6_5_w", "j_6_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_6_6_to_ax_6_5_w", "from": "j_6_6", "to": "ax_6_5_w", **get_kw("j_6_6", "ax_6_5_w")})
            elif c == 1 and r == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_1_5_to_ax_0_5_e", "from": "j_1_5", "to": "ax_0_5_e", **get_kw("j_1_5", "ax_0_5_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_0_5_e_to_j_1_5", "from": "ax_0_5_e", "to": "j_1_5", **get_kw("ax_0_5_e", "j_1_5")})
                ET.SubElement(edges_root, "edge", {"id": "ax_0_5_e_to_j_hosp3_w", "from": "ax_0_5_e", "to": "j_hosp3_w", **get_kw("ax_0_5_e", "j_hosp3_w")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_w_to_ax_0_5_e", "from": "j_hosp3_w", "to": "ax_0_5_e", **get_kw("j_hosp3_w", "ax_0_5_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_w_to_j_1_6", "from": "j_hosp3_w", "to": "j_1_6", **get_kw("j_hosp3_w", "j_1_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_1_6_to_j_hosp3_w", "from": "j_1_6", "to": "j_hosp3_w", **get_kw("j_1_6", "j_hosp3_w")})
            elif c == 2 and r == 5:
                ET.SubElement(edges_root, "edge", {"id": "j_2_5_to_j_hosp3_e", "from": "j_2_5", "to": "j_hosp3_e", **get_kw("j_2_5", "j_hosp3_e")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_e_to_j_2_5", "from": "j_hosp3_e", "to": "j_2_5", **get_kw("j_hosp3_e", "j_2_5")})
                ET.SubElement(edges_root, "edge", {"id": "j_hosp3_e_to_ax_2_5_w", "from": "j_hosp3_e", "to": "ax_2_5_w", **get_kw("j_hosp3_e", "ax_2_5_w")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_5_w_to_j_hosp3_e", "from": "ax_2_5_w", "to": "j_hosp3_e", **get_kw("ax_2_5_w", "j_hosp3_e")})
                ET.SubElement(edges_root, "edge", {"id": "ax_2_5_w_to_j_2_6", "from": "ax_2_5_w", "to": "j_2_6", **get_kw("ax_2_5_w", "j_2_6")})
                ET.SubElement(edges_root, "edge", {"id": "j_2_6_to_ax_2_5_w", "from": "j_2_6", "to": "ax_2_5_w", **get_kw("j_2_6", "ax_2_5_w")})
            else:
                n_bot = f"j_{c}_{r}"
                n_mid = f"ax_{c}_{r}_w"
                n_top = f"j_{c}_{r+1}"

                ET.SubElement(edges_root, "edge", {"id": f"{n_bot}_to_{n_mid}", "from": n_bot, "to": n_mid, **get_kw(n_bot, n_mid)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_bot}", "from": n_mid, "to": n_bot, **get_kw(n_mid, n_bot)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{n_top}", "from": n_mid, "to": n_top, **get_kw(n_mid, n_top)})
                ET.SubElement(edges_root, "edge", {"id": f"{n_top}_to_{n_mid}", "from": n_top, "to": n_mid, **get_kw(n_top, n_mid)})

                if c > 0 and c < (grid_cols - 1):
                    e_mid = f"ax_{c-1}_{r}_e"
                    ET.SubElement(edges_root, "edge", {"id": f"{n_mid}_to_{e_mid}", "from": n_mid, "to": e_mid, **get_kw(n_mid, e_mid)})
                    ET.SubElement(edges_root, "edge", {"id": f"{e_mid}_to_{n_mid}", "from": e_mid, "to": n_mid, **get_kw(e_mid, n_mid)})

    # Hospital center connections (Emergency access only)
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_n", "from": "hosp", "to": "j_hosp_n", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_n_to_hosp", "from": "j_hosp_n", "to": "hosp", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_s", "from": "hosp", "to": "j_hosp_s", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_s_to_hosp", "from": "j_hosp_s", "to": "hosp", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_e", "from": "hosp", "to": "j_hosp_e", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_e_to_hosp", "from": "j_hosp_e", "to": "hosp", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_to_j_hosp_w", "from": "hosp", "to": "j_hosp_w", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp_w_to_hosp", "from": "j_hosp_w", "to": "hosp", **hosp_edge_kwargs})

    # Hospital 2 center connections
    ET.SubElement(edges_root, "edge", {"id": "hosp_2_to_j_hosp2_n", "from": "hosp_2", "to": "j_hosp2_n", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp2_n_to_hosp_2", "from": "j_hosp2_n", "to": "hosp_2", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_2_to_j_hosp2_s", "from": "hosp_2", "to": "j_hosp2_s", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp2_s_to_hosp_2", "from": "j_hosp2_s", "to": "hosp_2", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_2_to_j_hosp2_e", "from": "hosp_2", "to": "j_hosp2_e", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp2_e_to_hosp_2", "from": "j_hosp2_e", "to": "hosp_2", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_2_to_j_hosp2_w", "from": "hosp_2", "to": "j_hosp2_w", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp2_w_to_hosp_2", "from": "j_hosp2_w", "to": "hosp_2", **hosp_edge_kwargs})

    # Hospital 3 center connections
    ET.SubElement(edges_root, "edge", {"id": "hosp_3_to_j_hosp3_n", "from": "hosp_3", "to": "j_hosp3_n", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp3_n_to_hosp_3", "from": "j_hosp3_n", "to": "hosp_3", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_3_to_j_hosp3_s", "from": "hosp_3", "to": "j_hosp3_s", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp3_s_to_hosp_3", "from": "j_hosp3_s", "to": "hosp_3", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_3_to_j_hosp3_e", "from": "hosp_3", "to": "j_hosp3_e", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp3_e_to_hosp_3", "from": "j_hosp3_e", "to": "hosp_3", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "hosp_3_to_j_hosp3_w", "from": "hosp_3", "to": "j_hosp3_w", **hosp_edge_kwargs})
    ET.SubElement(edges_root, "edge", {"id": "j_hosp3_w_to_hosp_3", "from": "j_hosp3_w", "to": "hosp_3", **hosp_edge_kwargs})

    with open(edg_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(edges_root))

    print(f"[SUCCESS] Generated expanded node file: {nod_filepath}")
    print(f"[SUCCESS] Generated expanded edge file: {edg_filepath}")

    # Compile with netconvert
    cmd = [
        "netconvert",
        f"--node-files={nod_filepath}",
        f"--edge-files={edg_filepath}",
        f"--output-file={net_filepath}",
        "--no-turnarounds=false"
    ]
    print(f"[INFO] Running netconvert: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SUCCESS] Successfully compiled expanded 3-hospital SUMO network: {net_filepath}")
    else:
        print(f"[ERROR] netconvert failed: {res.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    build_expanded_network()
