#!/usr/bin/env python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def writeplyfile(writefile, tet_nodes, tet_tot):
    """
    Writes a surface mesh in PLY format (ASCII).
    Indices in 'tet_tot' are expected to be 1-based.
    """
    in_size = tet_tot.shape
    if len(in_size) == 1:
        if in_size[0] > 0:
            tet_tot = tet_tot.reshape((1, in_size[0]))

    FILE = open(writefile, "w")
    FILE.write('ply \n')
    FILE.write('format ascii 1.0 \n')
    FILE.write('comment this is a surface \n')
    FILE.write('element vertex %i \n' % (tet_nodes.shape[0]))
    FILE.write('property float x \n')
    FILE.write('property float y \n')
    FILE.write('property float z \n')
    FILE.write('element face %i \n' % (tet_tot.shape[0]))
    FILE.write('property list uchar int vertex_index \n')
    FILE.write('end_header \n')

    # Nodes
    for i in range(len(tet_nodes)):
        x = tet_nodes[i, 0]
        y = tet_nodes[i, 1]
        z = tet_nodes[i, 2]
        FILE.write('%f %f %f\n' % (x, y, z))

    # Faces (converted to 0-based indexing)
    for i in range(len(tet_tot)):
        a1 = tet_tot[i, 0] - 1
        a2 = tet_tot[i, 1] - 1
        a3 = tet_tot[i, 2] - 1

        if in_size[1] == 3:
            FILE.write('3 %i %i %i\n' % (a1, a2, a3))
        elif in_size[1] == 4:
            a4 = tet_tot[i, 3] - 1
            FILE.write('4 %i %i %i %i\n' % (a1, a2, a3, a4))

    FILE.close()
    return

def make_triangle_connection(patch):
    """
    Creates the triangle connectivity between consecutive rings (slices).
    patch["height"] = number of slices
    patch["width"] = number of points in each ring
    Returns a (N, 3) array of 0-based indices.
    """
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

def calculate_normals(points, faces, node_id=None):
    """
    Computes cross-product normals for each face.
    If node_id is provided, returns the normal at that node by summing 
    the normals of adjacent faces.
    """
    face_normals = np.cross(
        points[faces[:, 1], :] - points[faces[:, 0], :],
        points[faces[:, 2], :] - points[faces[:, 0], :]
    )

    if node_id is None:
        return face_normals
    else:
        ix, iy = np.where(faces == node_id)
        face_normal_at_node = np.sum(face_normals[ix, :], axis=0)
        normal_length = np.sqrt((face_normal_at_node ** 2).sum(-1))
        if normal_length != 0.0:
            face_normal_at_node /= normal_length
        return face_normal_at_node

def cover_apex(nodes_renum, tris, patch, principal_axis=0):
    """
    Closes one end by creating an apex node and connecting it with the first ring of points.
    """
    nodes_covered = np.zeros((len(nodes_renum) + 1, 3))
    n_in_strip = patch["width"]
    tris_covered = np.zeros((len(tris) + n_in_strip, 3))

    # Example apex: average of the first ring
    p_coor = np.mean(nodes_renum[:patch["width"], :], axis=0)
    p_coor[principal_axis] += 0.0
    nodes_covered[1:, :] = nodes_renum
    nodes_covered[0, :] = p_coor

    # Shift faces by +1 (new apex node at index 0)
    tris_covered[n_in_strip:, :] = tris + 1

    # Create triangles connecting apex to the ring
    for i in range(n_in_strip):
        tris_covered[i, :] = [0, i + 1, (i + 1) % n_in_strip + 1]

    return nodes_covered, tris_covered

def cover_both_ends_centered(nodes, tris, patch, principal_axis=2):
    """
    Fecha ambas as extremidades triangulando radialmente
    a partir do centróide, ordenando os índices globais para
    evitar cruzamentos.
    - `nodes`: (M×3) array dos nós originais
    - `tris`: (K×3) conectividade entre fatias
    - patch["width"] = número de pontos por fatia
    - patch["height"] = número de fatias
    Retorna (nodes_covered, tris_final)
    """

    n_in_strip = patch["width"]
    n_slices   = patch["height"]

    # calcular índices globais das duas bordas
    base_idx = np.arange(0, n_in_strip)
    top_idx  = np.arange((n_slices-1)*n_in_strip,
                         n_slices*n_in_strip)

    # extrair suas coordenadas
    pts_base = nodes[base_idx]
    pts_top  = nodes[top_idx]

    # centroids
    cent_base = pts_base.mean(axis=0)
    cent_top  = pts_top.mean(axis=0)

    # novo array de nós: [cent_base, cent_top, nós_originais]
    nodes_cov = np.vstack([
        cent_base[np.newaxis,:],
        cent_top[np.newaxis,:],
        nodes
    ])

    # ajusta todos os índices de tris em +2
    tris_shift = tris + 2

    # função helper que devolve fan de triângulos
    def make_fan(cent_id, ring_idx):
        # calcular ângulos e ordenar os índices globais do ring
        rel = nodes[ring_idx] - nodes_cov[cent_id]
        ang = np.arctan2(rel[:,1], rel[:,0])
        order = ring_idx[np.argsort(ang)]
        # gerar triângulos
        fans = []
        n = len(order)
        for i in range(n):
            a = order[i]   + 2  # +2 pelo deslocamento de nós_cov
            b = order[(i+1)%n] + 2
            fans.append([cent_id, a, b])
        return np.array(fans, dtype=int)

    # centroid global indices em nodes_cov
    cent_base_id = 0
    cent_top_id  = 1

    # gera fans
    tris_base = make_fan(cent_base_id, base_idx)
    tris_top  = make_fan(cent_top_id,  top_idx)

    # concatena: base fan + lateral + top fan
    tris_final = np.vstack([tris_base, tris_shift, tris_top])

    return nodes_cov, tris_final


def replicate_single_slice_below(points, slice_thickness, principal_axis=2):
    """
    If there is only one slice (a single ring of points),
    replicate it "below" by 'slice_thickness' along 'principal_axis'.
    
    Returns:
      new_points: shape (2*N, 3), with:
          - the lower ring (original ring shifted downward)
          - the original ring at the original coords
      patch: {"height": 2, "width": N}
    """
    N = points.shape[0]
    
    # Create a copy of the ring, shifting by -slice_thickness
    ring_lower = points.copy()
    
    # If the sign is inverted, reverse the extrusion direction
    ring_lower[:, principal_axis] += slice_thickness 
    
    # Stack them: lower ring first, then the original ring
    new_points = np.vstack([ring_lower, points])
    
    # Now we have 2 slices, each with N points
    patch = {"height": 2, "width": N}
    return new_points, patch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate surface from points. Default: only one end is closed.")
    parser.add_argument("input_file", help="Text file with point coordinates (X Y Z).")
    parser.add_argument("--cover-both-ends", action="store_true",
                        help="Close both ends of the geometry instead of just one apex.")
    parser.add_argument("--slice-thickness", type=float, default=2.0,
                        help="Distance used if there's only one slice (default=2.0).")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    filename_input = args.input_file
    cover_both = args.cover_both_ends

    if not os.path.exists(filename_input):
        print(f"Error: File {filename_input} does not exist.")
        sys.exit(1)

    # Create output directory for .ply files
    output_dir = "./output/plyFiles"
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    filename_base = os.path.splitext(os.path.basename(filename_input))[0]
    filename_output = os.path.join(output_dir, f"{filename_base}.ply")

    # Invert coordinates
    invert_z = False
    invert_x = False
    intert_y = True

    user_input = {
        "print_ply": True,
        "principal_axis": 2,  # Z-axis
        "reshuffle_point_order": True,
        "cover_apex": not cover_both,
        "plot": False
    }

    print(f"Reading points from {filename_input}")
    points0 = np.loadtxt(filename_input)

    # Optionally invert Z
    if invert_z and len(points0) > 0:
        points0[:, 2] = -points0[:, 2]

    # Optionally invert X
    if invert_x and len(points0) > 0:
        points0[:, 0] = -points0[:, 0]

    # Optionally invert Y
    if intert_y and len(points0) > 0:
        points0[:, 1] = -points0[:, 1]
        
    # Optionally reverse the point order
    if user_input["reshuffle_point_order"]:
        points = points0[::-1, :]
    else:
        points = points0

    principal_axis = user_input["principal_axis"]
    # Attempt to discover how many slices
    slice_position_test = points[0, principal_axis]
    n_per_slice = np.sum(points[:, principal_axis] == slice_position_test)
    n_slices = int(len(points) / n_per_slice)

    if n_slices * n_per_slice != len(points):
        print("Error: Inconsistent number of points per slice.")
        sys.exit(1)

    # If there's only one slice, replicate it below
    if n_slices == 1:
        print("Detected a single slice. Replicating below using 'slice_thickness'...")
        new_points, patch = replicate_single_slice_below(
            points,
            slice_thickness=args.slice_thickness,
            principal_axis=principal_axis
        )
        points = new_points
        n_slices = patch["height"]
        n_per_slice = patch["width"]
    else:
        # If there's more than one slice, build patch normally
        patch = {"height": n_slices, "width": n_per_slice}

    # Create triangle surface
    tris = make_triangle_connection(patch)

    apex_first = None
    apex_last = None

    if cover_both:
        nodes_final, tris_final = cover_both_ends_centered(points, tris, patch, principal_axis=principal_axis)

    elif user_input["cover_apex"]:
        nodes_final, tris_final = cover_apex(points, tris, patch, principal_axis=principal_axis)
    else:
        nodes_final = points
        tris_final = tris

    # Calculate normals and remove degenerate triangles
    normals = calculate_normals(nodes_final, tris_final.astype(int))
    err_tol = 1e-6

    bad_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1))) < err_tol)[0]
    bad_indices = tris_final[bad_tris, :].astype(int).flatten()
    nodes_to_check = nodes_final[bad_indices, :]
    new_indices = bad_indices.copy()
    for i in range(len(bad_indices) - 1):
        for j in range(i + 1, len(bad_indices)):
            this_dist = np.sqrt(np.sum((nodes_to_check[i, :] - nodes_to_check[j, :]) ** 2))
            if this_dist < err_tol:
                new_indices[j] = new_indices[i]

    good_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1))) > err_tol)[0]
    tris_final = tris_final[good_tris, :]

    # Fix node indices if necessary
    tris_final_temp = tris_final.flatten()
    for i in range(tris_final.size):
        node_i = tris_final_temp[i]
        if node_i in bad_indices:
            if node_i not in new_indices:
                indice_i = np.where(node_i == bad_indices)[0][0]
                tris_final_temp[i] = new_indices[indice_i]
    tris_final = tris_final_temp.reshape(tris_final.shape)

    # Write .ply file
    if user_input["print_ply"]:
        print(f"Saving .ply file to {filename_output}")
        writeplyfile(filename_output, nodes_final, tris_final + 1)

    # Optional plot
    if user_input["plot"]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", label="Input Points")
        plt.legend()
        plt.show()
def triangulate_ring_to_center(center, ring_points, offset=0):
    """
    Ordena os pontos da borda em torno do centro e gera triângulos ligando ao centro.
    offset é o índice que será somado aos pontos (ex: para encaixar na lista geral)
    """
    # Translada pontos para o centro
    rel_points = ring_points - center
    angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
    order = np.argsort(angles)

    tris = []
    n = len(ring_points)
    for i in range(n):
        i0 = order[i]
        i1 = order[(i + 1) % n]
        tris.append([0, i0 + offset, i1 + offset])  # centro é o 0 no sistema local

    return np.array(tris)
