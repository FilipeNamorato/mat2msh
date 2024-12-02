#!/usr/bin/env python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def writeplyfile(writefile, tet_nodes, tet_tot):
    """Write triangularized mesh to a .ply file (ASCII format)."""
    in_size = tet_tot.shape

    # Open file for writing
    with open(writefile, "w") as FILE:
        FILE.write('ply\n')
        FILE.write('format ascii 1.0\n')
        FILE.write('comment this is a surface\n')
        FILE.write(f'element vertex {tet_nodes.shape[0]}\n')
        FILE.write('property float x\nproperty float y\nproperty float z\n')
        FILE.write(f'element face {tet_tot.shape[0]}\n')
        FILE.write('property list uchar int vertex_index\n')
        FILE.write('end_header\n')

        # Write vertices
        for x, y, z in tet_nodes:
            FILE.write(f'{x} {y} {z}\n')

        # Write faces
        for face in tet_tot:
            FILE.write(f'3 {face[0] - 1} {face[1] - 1} {face[2] - 1}\n')


def make_triangle_connection(patch):
    """Create triangular connections between slices."""
    n_strips = patch["height"] - 1
    n_in_strip = patch["width"]
    tris = np.zeros((n_strips * n_in_strip * 2, 3), dtype=int)

    for i in range(n_in_strip):
        for j in range(n_strips):
            i00 = i + n_in_strip * j
            i01 = i + n_in_strip * (j + 1)
            i11 = (i + 1) % n_in_strip + n_in_strip * (j + 1)
            i10 = (i + 1) % n_in_strip + n_in_strip * j

            tris[2 * i + j * n_in_strip * 2, :] = [i00, i01, i11]
            tris[2 * i + 1 + j * n_in_strip * 2, :] = [i00, i11, i10]

    return tris


def calculate_normals(points, faces):
    """Calculate normals of triangle faces."""
    return np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 2:
        print("Usage: python make_surface.py <input.txt>")
        sys.exit(1)

    filename_input = sys.argv[1]
    if not os.path.exists(filename_input):
        print(f"Error: File {filename_input} does not exist.")
        sys.exit(1)

    # Create output directory for .ply files
    output_dir = "./saida/plyFiles"
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    filename_base = os.path.splitext(os.path.basename(filename_input))[0]
    filename_output = os.path.join(output_dir, f"{filename_base}.ply")

    user_input = {
        "print_ply": True,
        "principal_axis": 2,
        "reshuffle_point_order": True,
        "cover_apex": True,
        "plot": False,
    }

    points = np.loadtxt(filename_input)

    if user_input["reshuffle_point_order"]:
        points = points[::-1, :]

    principal_axis = user_input["principal_axis"]
    n_per_slice = np.sum(points[:, principal_axis] == points[0, principal_axis])
    n_slices = len(points) // n_per_slice

    if n_slices * n_per_slice != len(points):
        print("Error: Inconsistent number of points per slice.")
        sys.exit(1)

    patch = {"height": n_slices, "width": n_per_slice}
    tris = make_triangle_connection(patch)

    nodes_final, tris_final = points, tris

    if user_input["print_ply"]:
        print(f"Saving .ply file to {filename_output}")
        writeplyfile(filename_output, nodes_final, tris_final + 1)

    if user_input["plot"]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", label="Input Points")
        plt.legend()
        plt.show()
