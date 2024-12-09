#!/usr/bin/env python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def writeplyfile(writefile,tet_nodes, tet_tot):
    """ Write triangularized mesh 0-indexed from input data 1-indexed"""
    in_size = tet_tot.shape
    # If only one triangle/quad
    if len(in_size)==1:
        if in_size[0]>0:
            tet_tot=tet_tot.reshapimport((1,in_size[0]))
             
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


def calculate_normals(points,faces,node_id=None):
    """ Calculate unit length normals of triangle faces or one point"""
    face_normals=np.cross( points[faces[:,1],:]-points[faces[:,0],:],
                           points[faces[:,2],:]-points[faces[:,0],:] )
 
    if node_id is None:
        #face_normals /= np.sqrt((face_normals ** 2).sum(-1))[..., np.newaxis]
        return face_normals
    else:
        ix,iy=np.where(faces==node_id)
        face_normal_at_node = np.sum(face_normals[ix,:],axis=0)
        normal_length = np.sqrt((face_normal_at_node ** 2).sum(-1))#[..., np.newaxis]
        if normal_length != 0.0:
            face_normal_at_node /= normal_length
         
        return face_normal_at_node

def cover_apex(nodes_renum, tris, patch,principal_axis=0):
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

    points0 = np.loadtxt(filename_input)

    # Reshuffle point order
    if user_input["reshuffle_point_order"]:
        points = points0[::-1,:]
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

    # Cover apex hole
    if user_input["cover_apex"]:
        nodes_final, tris_final = cover_apex(points, tris, patch,principal_axis=principal_axis)
    else:
        nodes_final = points
        tris_final = tris

    # Check data for degenerate tris
    normals= calculate_normals(nodes_final,tris_final.astype(int),node_id=None)
    err_tol = 0.000001
     
    # Removes duplicate indices (unfinished) 
    bad_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1)))<err_tol)[0]
    bad_indices = tris_final[bad_tris,:].astype(int).flatten()
    nodes_to_check = nodes_final[bad_indices,:]
    new_indices = bad_indices.copy()
    for i in range(len(bad_indices)-1):
        for j in range(i+1,len(bad_indices)):
            this_dist = np.sqrt(np.sum((nodes_to_check[i,:]-nodes_to_check[j,:])**2))
            if this_dist<err_tol:
                new_indices[j] = new_indices[i]
     
    #tris_final[bad_tris,:] = new_indices.reshape((len(bad_indices)/3,3))
    #nodes_final[bad_indices,:] = nodes_final[new_indices,:]
     
    good_tris = np.where(np.abs(np.sqrt((normals ** 2).sum(-1)))>err_tol)[0]
     
    # Remove bad tris
    tris_final = tris_final[good_tris,:]
 
    # Remove bad nodes
    tris_final_temp = tris_final.flatten()
    for i in range(tris_final.size):
        node_i = tris_final_temp[i]
        if node_i in bad_indices:
            if node_i not in new_indices:
                indice_i = np.where(node_i == bad_indices)[0][0]
                tris_final_temp[i] = new_indices[indice_i]
    tris_final = tris_final_temp.reshape(tris_final.shape)

    if user_input["print_ply"]:
        print(f"Saving .ply file to {filename_output}")
        writeplyfile(filename_output, nodes_final, tris_final + 1)

    if user_input["plot"]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", label="Input Points")
        plt.legend()
        plt.show()
