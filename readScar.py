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
    Lê os ROIs do arquivo .mat e retorna:
      - fatias: dict { z_val: [(x, y), (x, y), ...], ... }
      - pontos_3d: np.array shape (N, 3), para debug
    """
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)
    setstruct = data['setstruct']
    rois = setstruct[0][0]['Roi']

    fatias = {}  # { z: [(x, y), (x, y), ...] }

    for idx, roi in enumerate(rois):
        print(f"Processando ROI {idx+1}...")
        try:
            x_coords = roi['X'] if 'X' in roi.dtype.names else None
            y_coords = roi['Y'] if 'Y' in roi.dtype.names else None
            z_values = roi['Z'] if 'Z' in roi.dtype.names else None

            if x_coords is None or y_coords is None or z_values is None:
                print(f"Erro: Dados incompletos para ROI {idx+1}")
                continue

            for i in range(len(z_values)):
                x_arr = x_coords[i].flatten()
                y_arr = y_coords[i].flatten()
                z_val = float(z_values[i][0][0])

                if len(x_arr) == 0 or len(y_arr) == 0:
                    print(f"Erro: Fatia {z_val} do ROI {idx+1} está vazia.")
                    continue

                if z_val not in fatias:
                    fatias[z_val] = []
                
                # Adiciona os pontos (x, y) nesta fatia
                fatias[z_val].extend(zip(x_arr, y_arr))

                print(f"Adicionado ROI {idx+1}, Fatia {z_val}, "
                      f"Pontos={len(x_arr)}")

        except Exception as e:
            print(f"Erro ao processar ROI {idx+1}: {e}")

    # Cria um array unificado para debug
    pontos_3d = []
    for z, coords in fatias.items():
        for (x, y) in coords:
            pontos_3d.append([x, y, z])
    pontos_3d = np.array(pontos_3d)

    # (opcional) Salva os pontos originais
    np.savetxt("fibrosis_original.txt", pontos_3d)


    return fatias, pontos_3d

################################
# 2) Aplica deslocamentos e salva as fatias em .txt
################################
def save_fatias_to_txt(fatias, output_dir="fatias"):
    """
    Salva coordenadas X, Y das fibroses em arquivos TXT separados por fatia (Z),
    aplicando o deslocamento do alinhamento do miocárdio.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carrega deslocamentos
    try:
        endo_shifts_x = np.loadtxt("endo_shifts_x.txt")
        endo_shifts_y = np.loadtxt("endo_shifts_y.txt")
    except Exception as e:
        print(f"Erro ao carregar deslocamentos: {e}")
        return

    for z, coordenadas in fatias.items():
        filename = os.path.join(output_dir, f"fatia_{int(z)}.txt")
        print(f"Salvando fatia {z} no arquivo: {filename}")

        with open(filename, "w") as file:
            slice_idx = int(z)  # assumindo z como índice inteiro
            if 0 <= slice_idx < len(endo_shifts_x):
                shift_x = endo_shifts_x[slice_idx]
                shift_y = endo_shifts_y[slice_idx]
            else:
                shift_x = 0
                shift_y = 0
                print(f"Atenção: Fatia {slice_idx} fora do intervalo dos deslocamentos!")

            for x, y in coordenadas:
                x_aligned = x - shift_x
                y_aligned = y - shift_y
                file.write(f"{x_aligned} {y_aligned} {z}\n")

    print("Arquivos de fatias gerados com sucesso!")


################################
# 3) DBSCAN 2D por fatia
################################

def cluster_fibrosis_by_slice(fatias, eps_2d=2.0, min_samples_2d=5):
    """
    Aplica DBSCAN 2D para cada fatia.
    Retorna { z: { cluster_id: [(x,y), (x,y), ...] }, ... }
    """
    clusters_2d_por_fatia = defaultdict(dict)
    
    for z, coords in fatias.items():
        if not coords:
            continue
        coords_array = np.array(coords)
        if len(coords_array) < min_samples_2d:
            # Cada ponto vira cluster isolado
            for i, (x, y) in enumerate(coords_array):
                clusters_2d_por_fatia[z][i] = [(x,y)]
            continue

        db_2d = DBSCAN(eps=eps_2d, min_samples=min_samples_2d).fit(coords_array)
        labels_2d = db_2d.labels_

        # unique_labels contém todos os labels de clusters, exceto -1 (outliers)
        unique_labels = set(labels_2d) - {-1}
        for lbl in unique_labels:
            clusters_2d_por_fatia[z][lbl] = []

        for i, lbl in enumerate(labels_2d):
            if lbl == -1:
                continue
            x, y = coords_array[i]
            clusters_2d_por_fatia[z][lbl].append((x, y))

    return dict(clusters_2d_por_fatia)

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



def plot_clusters_2d_por_fatia(clusters_2d_por_fatia):
    for z, clusters in sorted(clusters_2d_por_fatia.items()):
        plt.figure(figsize=(6, 6))
        for cid, pontos in clusters.items():
            pts = np.array(pontos)
            plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cid}")
        plt.title(f"Fatia Z = {z}")
        plt.gca().invert_yaxis()  # MRI costuma ter Y invertido
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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

    # 2) Salvar fatias em .txt (opcional, para debug)
    save_fatias_to_txt(fatias_data, "fatias_txt")

    # 3) Clusterizar em 2D cada fatia
    clusters_2d_por_fatia = cluster_fibrosis_by_slice(
        fatias_data, 
        eps_2d=2, 
        min_samples_2d=1
    )

    #PLOTAR CLUSTER 2D POR FATIA
    plot_clusters_2d_por_fatia(clusters_2d_por_fatia)

    # 4) Calcular baricentros
    clusters_2d_list = compute_centroids_2d(clusters_2d_por_fatia)

    # 5) Conectar em 3D via baricentros
    clusters_3d = connect_2d_clusters_in_3d(
        clusters_2d_list, 
        base_radius_3d=1, 
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
