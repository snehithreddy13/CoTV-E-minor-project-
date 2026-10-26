import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def reconstruct_network():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    routes_file = os.path.join(script_dir, "routes.rou.xml")
    nod_filepath = os.path.join(script_dir, "network.nod.xml")
    edg_filepath = os.path.join(script_dir, "network.edg.xml")

    # Parse routes.rou.xml to extract all edge IDs
    tree = ET.parse(routes_file)
    root = tree.getroot()

    all_edges = set()
    for route in root.findall("route"):
        edges_str = route.get("edges", "")
        for e in edges_str.split():
            all_edges.add(e)

    print(f"Extracted {len(all_edges)} unique edge IDs from routes.rou.xml")

    nodes = set()
    edge_defs = []

    # Helper function to derive node coordinates
    def get_node_coords(node_id: str):
        if node_id == "hosp":
            return 1500.0, 1500.0
        elif node_id == "j_hosp_n":
            return 1500.0, 1600.0
        elif node_id == "j_hosp_s":
            return 1500.0, 1400.0
        elif node_id == "j_hosp_e":
            return 1600.0, 1500.0
        elif node_id == "j_hosp_w":
            return 1400.0, 1500.0

        m_j = re.match(r"^j_(\d+)_(\d+)$", node_id)
        if m_j:
            c, r = int(m_j.group(1)), int(m_j.group(2))
            return c * 600.0, r * 600.0

        m_ax = re.match(r"^ax_(\d+)_(\d+)_([n|s|e|w])$", node_id)
        if m_ax:
            c, r, d = int(m_ax.group(1)), int(m_ax.group(2)), m_ax.group(3)
            x, y = c * 600.0, r * 600.0
            if d == 'n':
                return x + 300.0, y + 20.0
            elif d == 's':
                return x + 300.0, y - 20.0
            elif d == 'e':
                return x + 20.0, y + 300.0
            elif d == 'w':
                return x - 20.0, y + 300.0

        return 0.0, 0.0

    # Parse edge endpoints from edge_id: "from_to_to"
    for e_id in all_edges:
        parts = e_id.split("_to_")
        if len(parts) == 2:
            src, dst = parts[0], parts[1]
            nodes.add(src)
            nodes.add(dst)
            edge_defs.append((e_id, src, dst))
            rev_id = f"{dst}_to_{src}"
            edge_defs.append((rev_id, dst, src))

    # Add hospital connection edges explicitly
    hospital_edges = [
        ("hosp_to_j_hosp_n", "hosp", "j_hosp_n"),
        ("j_hosp_n_to_hosp", "j_hosp_n", "hosp"),
        ("hosp_to_j_hosp_s", "hosp", "j_hosp_s"),
        ("j_hosp_s_to_hosp", "j_hosp_s", "hosp"),
        ("hosp_to_j_hosp_e", "hosp", "j_hosp_e"),
        ("j_hosp_e_to_hosp", "j_hosp_e", "hosp"),
        ("hosp_to_j_hosp_w", "hosp", "j_hosp_w"),
        ("j_hosp_w_to_hosp", "j_hosp_w", "hosp"),
    ]
    for e_id, src, dst in hospital_edges:
        nodes.add(src)
        nodes.add(dst)
        edge_defs.append((e_id, src, dst))

    # Remove duplicates from edge_defs
    unique_edge_defs = {}
    for eid, s, d in edge_defs:
        unique_edge_defs[eid] = (s, d)

    print(f"Total reconstructed nodes: {len(nodes)}")
    print(f"Total reconstructed edges: {len(unique_edge_defs)}")

    # Write network.nod.xml
    nodes_root = ET.Element("nodes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd"
    })

    for n_id in sorted(nodes):
        x, y = get_node_coords(n_id)
        ntype = "traffic_light" if n_id.startswith("j_") and not n_id.startswith("j_hosp") else "priority"
        ET.SubElement(nodes_root, "node", {
            "id": n_id,
            "x": f"{x:.1f}",
            "y": f"{y:.1f}",
            "type": ntype
        })

    with open(nod_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(nodes_root))

    # Write network.edg.xml
    edges_root = ET.Element("edges", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd"
    })

    edge_kwargs = {"numLanes": "2", "speed": "13.89", "priority": "3", "width": "16.0"}

    for e_id, (src, dst) in sorted(unique_edge_defs.items()):
        ET.SubElement(edges_root, "edge", {
            "id": e_id,
            "from": src,
            "to": dst,
            **edge_kwargs
        })

    with open(edg_filepath, "w", encoding="utf-8") as f:
        f.write(prettify(edges_root))

    print("[SUCCESS] Successfully reconstructed node and edge XML files from route definitions.")

if __name__ == "__main__":
    reconstruct_network()
