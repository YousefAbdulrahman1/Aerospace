import trimesh
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.spatial import KDTree

def remove_duplicate_nose(all_points, threshold=5e-2):
    """ Merge close points at the nose to avoid duplicate detections."""
    if all_points.shape[0] == 0:
        return all_points

    tree = KDTree(all_points)
    clusters = []
    visited = set()

    for i, point in enumerate(all_points):
        if i in visited:
            continue
        neighbors = tree.query_ball_point(point, threshold)
        cluster = np.mean(all_points[neighbors], axis=0)
        clusters.append(cluster)
        visited.update(neighbors)

    return np.array(clusters) if len(clusters) > 1 else all_points  # Ensure at least one point remains

def sort_points_clockwise(points):
    """ Sort points in a clockwise manner based on their angle from the centroid. """
    if points.shape[0] == 0:
        return points

    center = np.mean(points, axis=0)  # Compute centroid
    angles = np.arctan2(points[:, 2] - center[2], points[:, 1] - center[1])  # Compute angles relative to centroid
    sorted_indices = np.argsort(-angles)  # Sort in clockwise order
    return points[sorted_indices]

def save_intersection_to_xml(intersections, plane_origins, sampling_frequency, yz_center, output_file):
    root = ET.Element("explane", version="1.0")

    # Units section
    units = ET.SubElement(root, "Units")
    ET.SubElement(units, "length_unit_to_meter").text = "0.01"  # cm to meters
    ET.SubElement(units, "mass_unit_to_kg").text = "1"

    # Body section
    body = ET.SubElement(root, "body")
    ET.SubElement(body, "Name").text = "Body Name"

    color = ET.SubElement(body, "Color")
    ET.SubElement(color, "red").text = "98"
    ET.SubElement(color, "green").text = "102"
    ET.SubElement(color, "blue").text = "156"
    ET.SubElement(color, "alpha").text = "255"

    ET.SubElement(body, "Description").text = ""
    ET.SubElement(body, "Position").text = "0, 0, 0"
    ET.SubElement(body, "Type").text = "NURBS"
    ET.SubElement(body, "x_degree").text = "3"
    ET.SubElement(body, "hoop_degree").text = "3"
    ET.SubElement(body, "x_panels").text = "19"
    ET.SubElement(body, "hoop_panels").text = "11"

    inertia = ET.SubElement(body, "Inertia")
    ET.SubElement(inertia, "Volume_Mass").text = "0.000"

    # Frames for each cross-section
    for i, (plane_origin, intersection_segments) in enumerate(zip(plane_origins, intersections)):
        frame = ET.SubElement(body, "frame")
        ET.SubElement(frame, "Position").text = f"{plane_origin[0] * 0.1}, {yz_center[1] * 0.1}, {yz_center[0] * 0.1}"

        all_points = np.vstack(intersection_segments) if len(intersection_segments) > 0 else np.array([])
        all_points[:, [1, 2]] = all_points[:, [2, 1]]  # Swap Y and Z axes
        # all_points[:, 1] = -all_points[:, 1]  # Flip Y to correct mirroring
        if all_points.size > 0:
            all_points[:, [1, 2]] = all_points[:, [2, 1]]  # Swap Y and Z axes
            all_points[:, 1] -= yz_center[1]  # Center along new Y-axis
            all_points[:, 2] -= yz_center[0]  # Center along new Z-axis

            # Refine filtering to ensure precise division of right and left sides
            y_median = np.mean(all_points[:, 1])  # Compute precise center using mean instead of median
            all_points = all_points[all_points[:, 1] >= y_median]  # Ensure only right side points are kept

            if all_points.size > 0:
                # Sort points in clockwise order based on centroid
                all_points = sort_points_clockwise(all_points)

                # Attach first and last points to the XZ plane
                all_points[0, 1] = 0
                all_points[-1, 1] = 0

                sample_count = min(sampling_frequency, len(all_points))
                sampled_indices = np.linspace(0, len(all_points) - 1, sample_count, dtype=int)
                sampled_points = all_points[sampled_indices] * 0.1  # Scale down for XFLR5 compatibility

                for point in sampled_points:
                    ET.SubElement(frame, "point").text = f"{point[0]}, {point[1]}, {point[2]}"

    tree = ET.ElementTree(root)
    with open(output_file, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

def main(stl_file, plane_normal, section_frequency, sampling_frequency, cutoff_ratio, output_xml):
    mesh = trimesh.load_mesh(stl_file)
    bounds = mesh.bounds[:, 0]  # Get min and max along normal direction
    x_min, x_max = bounds[0], bounds[1]
    cutoff = (x_max - x_min) * cutoff_ratio  # Compute cutoff distance based on user input
    plane_positions = np.linspace(x_min + cutoff, x_max - cutoff, section_frequency)

    plane_origins = [
        np.array([pos if plane_normal[0] else 0,
                  pos if plane_normal[1] else 0,
                  pos if plane_normal[2] else 0])
        for pos in plane_positions
    ]

    intersections = [trimesh.intersections.mesh_plane(mesh, plane_normal, origin) for origin in plane_origins]

    # Compute yz_center (center of mass in Y and Z direction)
    all_points = np.vstack([seg for inter in intersections for seg in inter]) if intersections else np.array([])
    yz_center = np.mean(all_points[:, [2, 1]], axis=0) if all_points.size > 0 else np.array([0.0, 0.0])

    save_intersection_to_xml(intersections, plane_origins, sampling_frequency, yz_center, output_xml)

if __name__ == "__main__":
    stl_file = "part.STL"  # Change this to your STL file path
    plane_normal = np.array([1, 0, 0])  # Adjust for different orientations
    section_frequency = 50  # Number of sections along the object
    sampling_frequency = 15  # Adjust the number of radial samples per section
    cutoff_ratio = 0.02  # Adjust the cutoff percentage (e.g., 2% of the total length)
    output_xml = "fuselage.xml"
    main(stl_file, plane_normal, section_frequency, sampling_frequency, cutoff_ratio, output_xml)
