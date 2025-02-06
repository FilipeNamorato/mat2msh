#!/usr/bin/env python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def writeplyfile(writefile,tet_nodes, tet_tot):
    in_size = tet_tot.shape
    if len(in_size)==1:
        if in_size[0]>0:
            tet_tot=tet_tot.reshape((1,in_size[0]))  

    FILE=open(writefile,"w")
    FILE.write('ply \n')
    FILE.write('format ascii 1.0 \n')
    FILE.write('comment this is a surface \n')
    FILE.write('element vertex %i \n' % (tet_nodes.shape[0]) )
    FILE.write('property float x \n')
    FILE.write('property float y \n')
    FILE.write('property float z \n')
     
    FILE.write('element face %i \n' % (tet_tot.shape[0]) )
    FILE.write('property list uchar int vertex_index \n')
    FILE.write('end_header \n')
    # Nodes
    for i in range(len(tet_nodes)):
        x = tet_nodes[i,0]
        y = tet_nodes[i,1]
        z = tet_nodes[i,2]
         
        FILE.write('%f %f %f\n' % (x, y, z))
     
    # Faces NOTE: index from 0
    for i in range(len(tet_tot)):
        a1 = tet_tot[i,0]-1
        a2 = tet_tot[i,1]-1
        a3 = tet_tot[i,2]-1
         
        if in_size[1]==3:
            FILE.write('3 %i %i %i\n' % (a1,a2,a3))
        elif in_size[1]==4:
            a4 = tet_tot[i,3]-1
            FILE.write('4 %i %i %i %i\n' % (a1,a2,a3,a4))
     
    FILE.close()
    return

def make_triangle_connection(patch):
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

def calculate_normals(points,faces,node_id=None):
    face_normals=np.cross(
        points[faces[:,1],:] - points[faces[:,0],:],
        points[faces[:,2],:] - points[faces[:,0],:]
    )
    if node_id is None:
        return face_normals
    else:
        ix,iy=np.where(faces==node_id)
        face_normal_at_node = np.sum(face_normals[ix,:],axis=0)
        normal_length = np.sqrt((face_normal_at_node ** 2).sum(-1))
        if normal_length != 0.0:
            face_normal_at_node /= normal_length
        return face_normal_at_node

def cover_apex(nodes_renum, tris, patch, principal_axis=0):
    """
    (Mantido sem alterações)
    Fecha apenas um extremo (um 'apex').
    """
    nodes_covered = np.zeros((len(nodes_renum)+1,3))
    n_in_strip = patch["width"]
    tris_covered = np.zeros((len(tris)+n_in_strip,3))
     
    # Assume apex has largest principal axis coordinate
    p_coor = np.mean(nodes_renum[:patch["width"],:],axis=0)
    p_coor[principal_axis] += .0
    nodes_covered[1:,:] = nodes_renum
    nodes_covered[0,:] = p_coor
    tris_covered[n_in_strip:,:]=tris+1
    for i in range(n_in_strip):
        tris_covered[i,:] = [0,i+1,(i+1)%n_in_strip+1]
    return nodes_covered, tris_covered

def cover_both_ends(nodes_renum, tris, patch, principal_axis=0, apex_first=None, apex_last=None):
    """
    Fecha ambas as extremidades, adicionando 2 apexes.
    Permite usar apex_first e apex_last (se não forem None).
    """
    n_in_strip = patch["width"]
    n_slices   = patch["height"]
     
    # Se não for passado apex_first, calcula pela média do primeiro anel
    if apex_first is None:
        apex1 = np.mean(nodes_renum[:n_in_strip, :], axis=0)
        apex1[principal_axis] += 0.0
    else:
        apex1 = apex_first
     
    # Se não for passado apex_last, calcula pela média do último anel
    start_last_ring = (n_slices - 1)*n_in_strip
    if apex_last is None:
        apex2 = np.mean(nodes_renum[start_last_ring : start_last_ring + n_in_strip, :], axis=0)
        apex2[principal_axis] += 0.0
    else:
        apex2 = apex_last
     
    # Cria novos nós
    nodes_covered = np.zeros((len(nodes_renum) + 2, 3))
    nodes_covered[2:, :] = nodes_renum
    nodes_covered[0, :]  = apex1
    nodes_covered[1, :]  = apex2
     
    # Precisamos de +2*n_in_strip triângulos
    tris_covered = np.zeros((len(tris) + 2*n_in_strip, 3))
    tris_covered[2*n_in_strip:, :] = tris + 2
     
    # Fecha primeiro anel
    for i in range(n_in_strip):
        i0 = i + 2
        i1 = ((i + 1) % n_in_strip) + 2
        tris_covered[i, :] = [0, i0, i1]

    # Fecha último anel
    start_last_ring_new = start_last_ring + 2
    for i in range(n_in_strip):
        j0 = start_last_ring_new + i
        j1 = start_last_ring_new + ((i + 1) % n_in_strip)
        tris_covered[n_in_strip + i, :] = [1, j0, j1]

    return nodes_covered, tris_covered

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate surface from points. Default: only apex is closed.")
    parser.add_argument("input_file", help="Text file with point coordinates (X Y Z).")
    parser.add_argument("--cover-both-ends", action="store_true",
                        help="Close both ends of the geometry instead of just one apex.")
    return parser.parse_args()

def read_points_with_apex(filename):
    """
    Lê todos os pontos (X Y Z) de um arquivo texto.
    Se uma linha tiver '#', interpretamos as coordenadas da linha como apex.
    - O primeiro apex encontrado = apex_first
    - O último apex encontrado = apex_last
    Retorna (points, apex_first, apex_last).
    """
    points = []
    apex_first = None
    apex_last = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Se houver '#', quebrar antes do '#'
        if '#' in line_clean:
            line_no_comment = line_clean.split('#')[0].strip()
            coords = line_no_comment.split()
            if len(coords) < 3:
                continue
            x, y, z = map(float, coords[:3])
            # Se ainda não temos apex_first, define
            if apex_first is None:
                apex_first = np.array([x, y, z], dtype=float)
            # Sempre atualiza apex_last
            apex_last = np.array([x, y, z], dtype=float)
        else:
            # Linha normal
            coords = line_clean.split()
            if len(coords) < 3:
                continue
            x, y, z = map(float, coords)
            points.append([x, y, z])

    points_arr = np.array(points, dtype=float)
    return points_arr, apex_first, apex_last

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

    # >>> Boolean variable to invert Z if True <<<
    invert_z = True  # Ajuste para False se não quiser inverter o Z
    # >>> -------------------------------------- <<<

    user_input = {
        "print_ply": True,
        "principal_axis": 2,  # Eixo principal: Z
        "reshuffle_point_order": True,
        "cover_apex": not cover_both,
        "plot": False
    }

    print(f"Reading points from {filename_input}")
    points0, apex_first, apex_last = read_points_with_apex(filename_input)

    # Inverte coordenada Z se invert_z for True
    if invert_z and len(points0) > 0:
        points0[:, 2] = -points0[:, 2]
        if apex_first is not None:
            apex_first[2] = -apex_first[2]
        if apex_last is not None:
            apex_last[2] = -apex_last[2]

    if user_input["reshuffle_point_order"]:
        points = points0[::-1, :]
    else:
        points = points0

    principal_axis = user_input["principal_axis"]
    slice_position_test = points[0,principal_axis]
    n_per_slice = np.sum(points[:,principal_axis]==slice_position_test)
    n_slices = int(len(points)/n_per_slice)

    if n_slices * n_per_slice != len(points):
        print("Error: Inconsistent number of points per slice.")
        sys.exit(1)

    patch = {"height": n_slices, "width": n_per_slice}
    tris = make_triangle_connection(patch)

    # Decide se fecha as duas extremidades, apenas uma ou nenhuma
    if cover_both:
        # Fecha ambos os extremos, usando apex_first / apex_last se existirem
        nodes_final, tris_final = cover_both_ends(
            points, tris, patch,
            principal_axis=principal_axis,
            apex_first=apex_first,
            apex_last=apex_last
        )
    elif user_input["cover_apex"]:
        # Fecha somente um extremo (mantido original)
        nodes_final, tris_final = cover_apex(points, tris, patch, principal_axis=principal_axis)
    else:
        # Não fecha nada
        nodes_final = points
        tris_final = tris

    # Verifica triângulos degenerados
    normals = calculate_normals(nodes_final, tris_final.astype(int), node_id=None)
    err_tol = 0.000001

    bad_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1)))<err_tol)[0]
    bad_indices = tris_final[bad_tris,:].astype(int).flatten()
    nodes_to_check = nodes_final[bad_indices,:]
    new_indices = bad_indices.copy()
    for i in range(len(bad_indices)-1):
        for j in range(i+1,len(bad_indices)):
            this_dist = np.sqrt(np.sum((nodes_to_check[i,:]-nodes_to_check[j,:])**2))
            if this_dist<err_tol:
                new_indices[j] = new_indices[i]

    good_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1)))>err_tol)[0]
     
    # Remove triângulos degenerados
    tris_final = tris_final[good_tris,:]
 
    # Corrige índices de nós (se necessário)
    tris_final_temp = tris_final.flatten()
    for i in range(tris_final.size):
        node_i = tris_final_temp[i]
        if node_i in bad_indices:
            if node_i not in new_indices:
                indice_i = np.where(node_i == bad_indices)[0][0]
                tris_final_temp[i] = new_indices[indice_i]
    tris_final = tris_final_temp.reshape(tris_final.shape)

    # Salva arquivo .ply
    if user_input["print_ply"]:
        print(f"Saving .ply file to {filename_output}")
        writeplyfile(filename_output, nodes_final, tris_final + 1)

    # Plot (se habilitado)
    if user_input["plot"]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", label="Input Points")
        plt.legend()
        plt.show()
