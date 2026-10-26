import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def create_hospital_complex(root: ET.Element, hosp_id: str, center_x: float, center_y: float):
    """
    Creates a modern, multi-wing hospital complex with:
    - Main Medical Wings placed in four quadrants around the hospital junction
    - Asphalt-grey ambulance emergency bay / drop-off apron
    - Red Cross roof emblem
    - Soft green landscaping / courtyard
    Leaves a 40m clear buffer around all hospital access roads (centerlines at center_x and center_y).
    """
    # 1. Ambulance Apron / Parking Bay (Asphalt grey, layer 7)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_APRON",
        "type": "hospital_apron",
        "color": "70,80,95",
        "layer": "7",
        "shape": f"{center_x-110:.1f},{center_y-110:.1f} {center_x+110:.1f},{center_y-110:.1f} {center_x+110:.1f},{center_y+110:.1f} {center_x-110:.1f},{center_y+110:.1f}",
        "fill": "true"
    })

    # 2. Landscaping / Courtyard Lawn (Grass green, layer 8)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_LAWN",
        "type": "hospital_lawn",
        "color": "110,185,120",
        "layer": "8",
        "shape": f"{center_x-100:.1f},{center_y-100:.1f} {center_x+100:.1f},{center_y-100:.1f} {center_x+100:.1f},{center_y+100:.1f} {center_x-100:.1f},{center_y+100:.1f}",
        "fill": "true"
    })

    # 3. Main Hospital Building Wings (Medical Slate/Blue-White, layer 9)
    # Wing NW (x: center_x-95 to center_x-45, y: center_y+45 to center_y+95)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_WING_NW",
        "type": "hospital",
        "color": "225,238,248",
        "layer": "9",
        "shape": f"{center_x-95:.1f},{center_y+45:.1f} {center_x-45:.1f},{center_y+45:.1f} {center_x-45:.1f},{center_y+95:.1f} {center_x-95:.1f},{center_y+95:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "45"}))

    # Wing NE (x: center_x+45 to center_x+95, y: center_y+45 to center_y+95)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_WING_NE",
        "type": "hospital",
        "color": "225,238,248",
        "layer": "9",
        "shape": f"{center_x+45:.1f},{center_y+45:.1f} {center_x+95:.1f},{center_y+45:.1f} {center_x+95:.1f},{center_y+95:.1f} {center_x+45:.1f},{center_y+95:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "45"}))

    # Wing SW (x: center_x-95 to center_x-45, y: center_y-95 to center_y-45)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_WING_SW",
        "type": "hospital",
        "color": "225,238,248",
        "layer": "9",
        "shape": f"{center_x-95:.1f},{center_y-95:.1f} {center_x-45:.1f},{center_y-95:.1f} {center_x-45:.1f},{center_y-45:.1f} {center_x-95:.1f},{center_y-45:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "45"}))

    # Wing SE (x: center_x+45 to center_x+95, y: center_y-95 to center_y-45)
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_WING_SE",
        "type": "hospital",
        "color": "225,238,248",
        "layer": "9",
        "shape": f"{center_x+45:.1f},{center_y-95:.1f} {center_x+95:.1f},{center_y-95:.1f} {center_x+95:.1f},{center_y-45:.1f} {center_x+45:.1f},{center_y-45:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "45"}))

    # 4. Emergency Red Cross Roof Emblem on NW Wing (Red, layer 10)
    # Vertical cross bar
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_CROSS_V",
        "type": "hospital_emblem",
        "color": "220,35,35",
        "layer": "10",
        "shape": f"{center_x-73:.1f},{center_y+55:.1f} {center_x-67:.1f},{center_y+55:.1f} {center_x-67:.1f},{center_y+85:.1f} {center_x-73:.1f},{center_y+85:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "46"}))

    # Horizontal cross bar
    ET.SubElement(root, "poly", {
        "id": f"{hosp_id}_CROSS_H",
        "type": "hospital_emblem",
        "color": "220,35,35",
        "layer": "10",
        "shape": f"{center_x-85:.1f},{center_y+67:.1f} {center_x-55:.1f},{center_y+67:.1f} {center_x-55:.1f},{center_y+73:.1f} {center_x-85:.1f},{center_y+73:.1f}",
        "fill": "true"
    }).append(ET.Element("param", {"key": "height", "value": "46"}))

def generate_buildings_file(grid_cols: int, grid_rows: int, out_filename: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, out_filename)

    root = ET.Element("additional", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd"
    })

    # Curated modern building color palettes
    building_colors = [
        "225,140,90",   # Warm terracotta
        "240,225,190",  # Sand beige
        "120,180,220",  # Glass blue
        "190,160,210",  # Soft violet
        "90,170,210",   # Commercial cyan
        "200,200,210",  # Modern grey
        "135,206,235",  # Sky blue
        "144,238,144"   # Soft sage
    ]

    color_idx = 0

    # Hospitals to place:
    # Hospital 1: at (1500, 1500) between col 2-3, row 2-3
    # Hospital 2: at (3300, 3300) between col 5-6, row 5-6 (only in 8x8 grid)

    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            # Check if this cell is occupied by Hospital 1 (col 2, row 2)
            if c == 2 and r == 2:
                create_hospital_complex(root, "HOSPITAL_1", 1500.0, 1500.0)
                continue
            # Check if this cell is occupied by Hospital 2 (col 5, row 5) in 8x8 grid
            elif c == 5 and r == 5 and grid_cols >= 8:
                create_hospital_complex(root, "HOSPITAL_2", 3300.0, 3300.0)
                continue
            # Check if this cell is occupied by Hospital 3 (col 1, row 5) in 8x8 grid
            elif c == 1 and r == 5 and grid_cols >= 8:
                create_hospital_complex(root, "HOSPITAL_3", 900.0, 3300.0)
                continue

            # Regular City Block: Bounds x in [c*600, (c+1)*600], y in [r*600, (r+1)*600]
            # Setback: 40m buffer from road centerlines (roads run along x=c*600, x=(c+1)*600, y=r*600, y=(r+1)*600)
            x_min = c * 600.0 + 40.0
            x_max = (c + 1) * 600.0 - 40.0
            y_min = r * 600.0 + 40.0
            y_max = (r + 1) * 600.0 - 40.0

            x_mid = (x_min + x_max) / 2.0
            y_mid = (y_min + y_max) / 2.0

            # 4 Sub-Plots inside the Block: SW, SE, NW, NE
            plots = [
                (f"BLOCK_{c}_{r}_P1", x_min + 15, x_mid - 15, y_min + 15, y_mid - 15),
                (f"BLOCK_{c}_{r}_P2", x_mid + 15, x_max - 15, y_min + 15, y_mid - 15),
                (f"BLOCK_{c}_{r}_P3", x_min + 15, x_mid - 15, y_mid + 15, y_max - 15),
                (f"BLOCK_{c}_{r}_P4", x_mid + 15, x_max - 15, y_mid + 15, y_max - 15),
            ]

            for plot_id, px1, px2, py1, py2 in plots:
                col = building_colors[color_idx % len(building_colors)]
                color_idx += 1

                # Building Polygon
                p_elem = ET.SubElement(root, "poly", {
                    "id": plot_id,
                    "type": "building",
                    "color": col,
                    "layer": "9",
                    "shape": f"{px1:.1f},{py1:.1f} {px2:.1f},{py1:.1f} {px2:.1f},{py2:.1f} {px1:.1f},{py2:.1f}",
                    "fill": "true"
                })
                h_val = 30 + ((c + r) * 7) % 35
                p_elem.append(ET.Element("param", {"key": "height", "value": str(h_val)}))

                # Boundary Fence Polygon
                fx1, fx2 = px1 - 5.0, px2 + 5.0
                fy1, fy2 = py1 - 5.0, py2 + 5.0
                ET.SubElement(root, "poly", {
                    "id": f"{plot_id}_FENCE",
                    "type": "fence",
                    "color": "90,95,100",
                    "layer": "11",
                    "shape": f"{fx1:.1f},{fy1:.1f} {fx2:.1f},{fy1:.1f} {fx2:.1f},{fy2:.1f} {fx1:.1f},{fy2:.1f}",
                    "fill": "false"
                }).append(ET.Element("param", {"key": "height", "value": "2"}))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prettify(root))

    print(f"[SUCCESS] Generated building addition file: {out_path}")

def main():
    # Generate unified 8x8 multi-hospital buildings.add.xml
    generate_buildings_file(8, 8, "buildings.add.xml")

if __name__ == "__main__":
    main()
