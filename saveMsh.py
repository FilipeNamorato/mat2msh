import numpy as np
from scipy.io import loadmat
from scipy.spatial import Delaunay
import os
import meshio

def export_to_txt(mat_filename, output_prefix, voxel_depth=1.0, interpolation_factor=2):
    """
    Exporta coordenadas das estruturas do arquivo .mat para arquivos .txt,
    com interpolação para maior resolução e ajustando o eixo Z com base na profundidade do voxel.
    
    Parameters:
    - mat_filename: Caminho para o arquivo .mat com os dados alinhados.
    - output_prefix: Prefixo dos arquivos .txt gerados.
    - voxel_depth: Profundidade entre as fatias originais.
    - interpolation_factor: Número de camadas intermediárias para interpolação.
    """
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']

    structures = {
        'Endo': ('EndoX', 'EndoY'),
        'Epi': ('EpiX', 'EpiY'),
        'RVEndo': ('RVEndoX', 'RVEndoY'),
        'RVEpi': ('RVEpiX', 'RVEpiY')
    }

    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)

    for name, (x_attr, y_attr) in structures.items():
        try:
            x_coords = getattr(setstruct, x_attr)
            y_coords = getattr(setstruct, y_attr)

            if x_coords.ndim == 1:
                x_coords = x_coords[:, np.newaxis]
                y_coords = y_coords[:, np.newaxis]
                num_slices = 1
            elif x_coords.ndim == 2:
                num_slices = x_coords.shape[1]
            else:
                num_slices = x_coords.shape[2]

            # Interpolação entre camadas
            for s in range(num_slices - 1):
                x_slice = x_coords[:, 0, s] if x_coords.ndim == 3 else x_coords[:, s]
                y_slice = y_coords[:, 0, s] if x_coords.ndim == 3 else y_coords[:, s]
                next_x_slice = x_coords[:, 0, s + 1] if x_coords.ndim == 3 else x_coords[:, s + 1]
                next_y_slice = y_coords[:, 0, s + 1] if y_coords.ndim == 3 else y_coords[:, s + 1]

                for i in range(interpolation_factor):
                    alpha = i / interpolation_factor
                    interpolated_x = (1 - alpha) * x_slice + alpha * next_x_slice
                    interpolated_y = (1 - alpha) * y_slice + alpha * next_y_slice
                    z_value = (s + alpha) * voxel_depth

                    valid_mask = ~np.isnan(interpolated_x) & ~np.isnan(interpolated_y)
                    if np.any(valid_mask):
                        coords = np.column_stack((interpolated_x[valid_mask], interpolated_y[valid_mask], np.full(valid_mask.sum(), z_value)))
                        filename = f"{output_prefix}/{name}_slice{s}_interp_{i}.txt"
                        np.savetxt(filename, coords, delimiter=' ', header=f"{name} coordinates slice {s} interpolation {i}")
                        print(f"Arquivo salvo: {filename}")

        except AttributeError:
            print(f"Erro: {x_attr} ou {y_attr} não encontrado no arquivo {mat_filename}")

def generate_msh_with_delaunay(output_prefix, msh_filename):
    """
    Gera uma malha 3D em formato .msh a partir de arquivos .txt contendo coordenadas,
    utilizando triangulação Delaunay para conectar pontos adjacentes.
    """
    points = []

    # Lê os arquivos .txt e armazena as coordenadas
    for filename in sorted(os.listdir(output_prefix)):
        if filename.endswith(".txt"):
            coords = np.loadtxt(os.path.join(output_prefix, filename), delimiter=' ')
            if coords.size > 0:
                points.extend(coords[:, :3])

    points = np.array(points)

    # Triangulação Delaunay para conectar pontos adjacentes
    delaunay_tri = Delaunay(points[:, :2])  # Apenas X, Y para triangulação de superfície
    cells = [("triangle", delaunay_tri.simplices)]

    # Estrutura da malha com células triangulares para maior precisão
    mesh = meshio.Mesh(
        points=points,
        cells=cells
    )

    # Exporta o arquivo `.msh` no formato Gmsh 2.2 em texto ASCII
    meshio.write(msh_filename, mesh, file_format="gmsh22")
    print(f"Arquivo .msh salvo como: {msh_filename}")

# Exemplo de uso
export_to_txt("Patient_1.mat", "output_prefix", voxel_depth=1.0, interpolation_factor=3)
generate_msh_with_delaunay("output_prefix", "malha_3d_alinhada.msh")
