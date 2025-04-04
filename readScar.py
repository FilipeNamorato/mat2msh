import sys
import os
import glob
import subprocess
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.io import loadmat
from collections import defaultdict
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

################################
# 1) Lê o arquivo .mat e extrai as fatias + pontos 3D
################################
def readScar(mat_filename):
    """
    Lê o arquivo .mat, extrai cada ROI e armazena num dicionário:
      {
        z_val : {
          roi_name : [(x,y), (x,y), ...],
          ...
        },
        ...
      }
    Também retorna pontos_3d unificados apenas para debug/plot.
    """
    import numpy as np
    from scipy.io import loadmat

    print(f"Reading file (ROI-based): {mat_filename}")
    data = loadmat(mat_filename)
    setstruct = data['setstruct']
    rois = setstruct[0][0]['Roi']  # array de ROIs

    # dicionário: {z_val: {roi_name: [(x,y), (x,y), ...]}}
    fatias = {}
    pontos_3d = []

    for idx, roi in enumerate(rois):
        # 1) Lê o campo 'Name'
        #Precisamos de [0][0] para chegar na string real.
        #Ex.: se printar rois['Name'][0][0], obtenho ['ROI-1'] (array),
        #mas rois['Name'][0][0][0] => 'ROI-1' (str).
        
        #Como cada 'roi' é um elemento do array, faz-se roi['Name'][0][0] => array(['ROI-1']).
        #Então [0] adicional pega a string 'ROI-1' em si.
        roi_name_raw = roi['Name'][0][0][0]  # deve ser algo como 'ROI-1'
        roi_name = str(roi_name_raw).strip()

        # 2) Lê arrays de X, Y, Z (caso existam no dtype)
        x_coords = roi['X'] if 'X' in roi.dtype.names else None
        y_coords = roi['Y'] if 'Y' in roi.dtype.names else None
        z_vals   = roi['Z'] if 'Z' in roi.dtype.names else None

        if x_coords is None or y_coords is None or z_vals is None:
            print(f"[readScar] ROI {idx+1} ('{roi_name}') sem X/Y/Z. Pulando.")
            continue

        # 3) Itera sobre todas as sub-fatias desse ROI
        num_slices = len(z_vals)
        for i in range(num_slices):
            # Cada z_vals[i] normalmente é do tipo [[algum_valor]]
            z_i = int(z_vals[i][0][0])

            x_arr = x_coords[i].flatten()
            y_arr = y_coords[i].flatten()

            if len(x_arr) == 0 or len(y_arr) == 0:
                print(f"[readScar] ROI '{roi_name}' sub-fatia {i} está vazia.")
                continue

            # Monta lista de tuplas (x,y)
            coords_2d = list(zip(x_arr, y_arr))

            # 4) Guarda no dicionário, garantindo que cada ROI numa fatia é um cluster separado
            if z_i not in fatias:
                fatias[z_i] = {}
            if roi_name not in fatias[z_i]:
                fatias[z_i][roi_name] = []

            fatias[z_i][roi_name].extend(coords_2d)

            # 5) Também guarda em pontos_3d para debug
            for (xx, yy) in coords_2d:
                pontos_3d.append([xx, yy, z_i])

    # Converte em array (debug)
    pontos_3d = np.array(pontos_3d)

    # Salva debug (opcional)
    np.savetxt("fibrosis_original.txt", pontos_3d)

    return fatias, pontos_3d

################################
# 2) Aplica deslocamentos e salva as fatias em .txt
################################
def save_fatias_to_txt(fatias, output_dir="fatias"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Tenta carregar deslocamentos
    try:
        endo_shifts_x = np.loadtxt("endo_shifts_x.txt")
        endo_shifts_y = np.loadtxt("endo_shifts_y.txt")
    except Exception as e:
        print(f"Erro ao carregar deslocamentos: {e}")
        return

    # fatias = { z_val: { roi_name: [(x,y), ...], ... }, ... }
    for z, dict_clusters in fatias.items():
        filename = os.path.join(output_dir, f"fatia_{int(z)}.txt")
        print(f"Salvando fatia {z} no arquivo: {filename}")

        slice_idx = int(z)
        if 0 <= slice_idx < len(endo_shifts_x):
            shift_x = endo_shifts_x[slice_idx]
            shift_y = endo_shifts_y[slice_idx]
        else:
            shift_x = 0
            shift_y = 0

        with open(filename, "w") as file:
            # Percorre cada ROI dessa fatia
            for roi_name, coords_lista in dict_clusters.items():
                # coords_lista deve ser [(x1, y1), (x2, y2), ...]
                for (x, y) in coords_lista:
                    x_aligned = x - shift_x
                    y_aligned = y - shift_y
                    file.write(f"{x_aligned} {y_aligned} {z}\n")

    print("Arquivos de fatias gerados com sucesso!")
    
################################
# 4) Cálculo de baricentros
################################
def compute_centroids_2d(clusters_2d_por_fatia):
    """
    Gera uma lista de objetos representando cada cluster 2D, com:
      z, cluster_id, centroid, points
    """
    from collections import namedtuple
    Cluster2D = namedtuple("Cluster2D", ["z", "cluster_id", "centroid", "points"])

    result = []
    for z, clusters_dict in clusters_2d_por_fatia.items():
        for cid, pts_2d in clusters_dict.items():
            arr = np.array(pts_2d)
            centroid = arr.mean(axis=0)
            c2d = Cluster2D(z=z, cluster_id=cid, centroid=centroid, points=pts_2d)
            result.append(c2d)
    return result

################################
# 5) Conecta baricentros em 3D
################################
from scipy.spatial import distance
from collections import defaultdict
def min_distance_between_clusters(cluster1, cluster2):
    """
    Calcula a menor distância entre quaisquer dois pontos de dois clusters 2D.
    """
    points1 = np.array(cluster1.points)
    points2 = np.array(cluster2.points)

    # Verifica se ambos os clusters têm pontos válidos
    if points1.size == 0 or points2.size == 0:
        return float('inf')  # Retorna infinito se algum estiver vazio
    
    return np.min(distance.cdist(points1, points2, 'euclidean'))


def connect_2d_clusters_in_3d(clusters_2d_list, base_radius_3d=3.0, slice_thickness=1.0, max_delta_z=1):
    n = len(clusters_2d_list)
    adj = [[] for _ in range(n)]  # Lista de adjacência do grafo

    for i in range(n):
        for j in range(i+1, n):
            # Ignora clusters da mesma fatia
            if clusters_2d_list[i].z == clusters_2d_list[j].z:
                continue

            # Respeita o limite de diferença em Z
            if abs(clusters_2d_list[i].z - clusters_2d_list[j].z) > max_delta_z:
                continue

            d = min_distance_between_clusters(clusters_2d_list[i], clusters_2d_list[j])

            dynamic_radius_3d = max(
                base_radius_3d, 
                (len(clusters_2d_list[i].points) + len(clusters_2d_list[j].points)) * 0.05
            )

            if d < dynamic_radius_3d:
                adj[i].append(j)
                adj[j].append(i)

    # Algoritmo de DFS para encontrar componentes conexas
    visited = [False] * n
    componente = [-1] * n
    comp_id = 0

    def dfs(start):
        stack = [start]
        visited[start] = True
        componente[start] = comp_id
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    componente[v] = comp_id
                    stack.append(v)

    # Executar DFS para encontrar todas as componentes conexas
    for i in range(n):
        if not visited[i]:
            dfs(i)
            comp_id += 1

    # Montar clusters 3D
    clusters_3d = defaultdict(list)
    for i, c2d in enumerate(clusters_2d_list):
        cid = componente[i]
        for (x, y) in c2d.points:
            clusters_3d[cid].append((x, y, c2d.z))

    return dict(clusters_3d)

################################
# 6) Salvar clusters 3D em .txt
################################
def save_clusters_to_txt(clusters, mat_filename, output_dir="clusters_dbscan"):
    """
    Para cada cluster, salva em cluster_<id>.txt, escalando (x, y, z) pela resolução do .mat.
    """
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)
    setstruct = data['setstruct']

    slice_thickness = setstruct['SliceThickness'][0][0][0][0]
    slice_gap       = setstruct['SliceGap'][0][0][0][0]
    resolution_x    = setstruct['ResolutionX'][0][0][0][0]
    resolution_y    = setstruct['ResolutionY'][0][0][0][0]

    print("------------------------------------------------------")
    print("Resolutions:", resolution_x, resolution_y)
    print("------------------------------------------------------")

    os.makedirs(output_dir, exist_ok=True)

    for lbl, pts in clusters.items():
        filename = os.path.join(output_dir, f"cluster_{lbl}.txt")
        with open(filename, "w") as f:
            for (x, y, z) in pts:
                x_scaled = x * resolution_x
                y_scaled = y * resolution_y
                z_scaled = z * (slice_thickness + slice_gap)
                #print(f"Z: {z}, Z_scaled: {z_scaled}")
                #z_scaled = z * 1
                f.write(f"{x_scaled} {y_scaled} {z_scaled}\n")

        print(f"Cluster {lbl} saved to: {filename}")

################################
# MAIN: Pipeline completo
################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro: Nenhum arquivo .mat foi especificado.")
        print("Uso: python3 readScar.py <caminho_do_arquivo.mat>")
        sys.exit(1)

    mat_filename = sys.argv[1]

    # 1) Ler ROIs
    fatias_data, pontos_3d = readScar(mat_filename)

    print("Chaves de fatias_data:", sorted(fatias_data.keys()))
    for z_val, rois_dict in fatias_data.items():
        print(f"Z={z_val}, ROIs={list(rois_dict.keys())}")

    # 2) Salvar fatias em .txt (opcional, para debug)
    save_fatias_to_txt(fatias_data, "fatias_txt")

    # 4) Calcular baricentros
    clusters_2d_list = compute_centroids_2d(fatias_data)

    # 5) Conectar em 3D via baricentros
    clusters_3d = connect_2d_clusters_in_3d(
        clusters_2d_list, 
        base_radius_3d=0.5, 
        slice_thickness=1
    )

    # 6) Salvar clusters em TXT
    save_clusters_to_txt(clusters_3d, mat_filename, "clusters_dbscan")

    # 7) Gera superfícies .ply para cada cluster
    cluster_txt_files = sorted(glob.glob("./clusters_dbscan/cluster_*.txt"))
    for surface_file in cluster_txt_files:
        try:
            surface_command = f"python3 make_surface.py {surface_file} --cover-both-ends"
            subprocess.run(surface_command, shell=True, check=True)
            print(f"Superfície gerada para {surface_file}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao gerar superfície para {surface_file}: {e}")

    # 8) Converte cada .ply em .stl
    for surface_file in cluster_txt_files:
        cluster_name = os.path.splitext(os.path.basename(surface_file))[0]
        ply_file = f"./output/plyFiles/{cluster_name}.ply"
        stl_output = f"./output/scarFiles/{cluster_name}.stl"

        if not os.path.exists(ply_file):
            print(f"Erro: não existe arquivo PLY gerado para {ply_file}.")
            continue

        try:
            ply_to_stl_command = f"./convertPly2STL/build/PlyToStl {ply_file} {stl_output} 1"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            print(f"STL gerado com sucesso: {stl_output}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao converter {ply_file} para {stl_output}: {e}")
