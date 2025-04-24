# Pipeline de Extração e Processamento de Fibrose (ROI para Cicatriz em 3D)
# =============================================================================
# Este script realiza todo o fluxo de trabalho para extrair regiões de interesse (ROIs)
# de um arquivo .mat de ressonância magnética, aplicar alinhamento, agrupar pontos
# 2D por fatias, calcular centróides e conectar essas ROIs 2D em estruturas 3D (cicatrizes),
# além de gerar malhas de superfície e arquivos STL.
#
# Etapas principais:
# 1) Leitura do .mat e extração de ROIs: obtém nome, coordenadas (X,Y) e índice de fatia (Z).
# 2) Agrupamento por fatia: organiza pontos em um dicionário fatiass -> {z: {roi_name: [pts]}}.
# 3) Visualização 2D: plota cada fatia mostrando pontos e seus centróides para conferência.
# 4) Alinhamento e gravação de fatias: aplica deslocamentos (shifts) em X e Y
#    e grava arquivos .txt para cada fatia com coordenadas alinhadas.
# 5) Cálculo de centróides 2D: gera objetos com coordenadas médias por ROI em cada fatia.
# 6) Conexão 2D→3D: monta um grafo de adjacência entre centróides de fatias
#    consecutivas e encontra componentes conexas, formando clusters 3D.
# 7) Gravação de clusters 3D: salva cada cluster em .txt usando resolução e espessura
#    de fatia para dimensionar corretamente X, Y e Z.
# 8) Geração de superfícies e STL: cria malhas .ply e converte para STL para modelagem.
# =============================================================================
import sys, os, glob, subprocess, argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial import distance, KDTree
from sklearn.cluster import DBSCAN

# Estrutura simples para cada ROI em uma fatia: nome, índice Z e lista de pontos (x,y)
ROIEntry = namedtuple('ROIEntry', ['name', 'z', 'points'])

################################
# 1) Leitura do .mat e extração de ROIs
################################
def readScar(mat_filename):
    """
    Lê o arquivo .mat e retorna uma lista plana de ROIEntry.
    Cada ROIEntry contém:
      - name: nome da ROI (ex: "ROI-1")
      - z: índice da fatia onde essa ROI foi anotada
      - points: lista de tuplas (x,y) dos pontos da ROI

    Explicação:
    - "setstruct" é a estrutura MATLAB com metadados e ROIs.
    - Extraímos de cada ROI seus arrays X, Y, Z e o nome para cada elemento.
    - Para cada sub-fat ia (índice i), associamos o nome correto e empacotamos
      os pontos convertidos em Python.
    """
    print(f"Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename)
    rois = data['setstruct'][0][0]['Roi']

    entries = []
    for idx, roi in enumerate(rois):
        # Extrai lista de nomes para cada sub-fat ia
        raw_names = roi['Name'].flatten()
        slice_names = []
        for element in raw_names:
            # Converte array numpy para string
            if isinstance(element, np.ndarray):
                val = element.flat[0]
            else:
                val = element
            slice_names.append(str(val).strip())
        # Arrays de coordenadas
        X = roi['X']; Y = roi['Y']; Z = roi['Z']
        # Para cada sub-fatia, empacota um ROIEntry
        for i in range(len(Z)):
            z_val = int(np.atleast_1d(Z[i]).flat[0])
            x_arr = np.atleast_1d(X[i]).flatten()
            y_arr = np.atleast_1d(Y[i]).flatten()
            if x_arr.size == 0 or y_arr.size == 0:
                continue  # sem pontos nesta sub-fatia
            name = slice_names[i] if i < len(slice_names) else slice_names[0]
            pts = list(zip(x_arr, y_arr))
            entries.append(ROIEntry(name, z_val, pts))
    return entries

################################
# 2) Agrupamento de ROIs por fatia
################################
def group_by_slice(entries):
    """
    Recebe lista de ROIEntry e retorna:
      { z: { roi_name: [ (x,y), ... ], ... }, ... }
    """
    fatias = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias[e.z][e.name].extend(e.points)
    return fatias

################################
# 3) Visualização 2D das fatias
################################
def plot_slices(fatias):
    """
    Para conferência, plota cada fatia mostrando:
      - Pontos de cada ROI coloridos
      - Nome e centróide marcado
    """
    for z, roi_map in sorted(fatias.items()):
        plt.figure(figsize=(6,6))
        for name, pts in roi_map.items():
            arr = np.array(pts)
            plt.scatter(arr[:,0], arr[:,1], label=name, s=20)
            cen = arr.mean(axis=0)
            plt.text(cen[0], cen[1], name,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', alpha=0.5))
        plt.title(f"Slice Z={z}")
        plt.gca().invert_yaxis()
        plt.legend(fontsize=7)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

################################
# 4) Alinhamento e gravação de fatias em .txt
################################
def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir="fatias"):
    """
    Aplica deslocamentos (shifts) em X/Y para cada fatia e salva em "fatias/".
    Arquivos: fatia_<z>.txt contendo linhas "x_alinhado y_alinhado z".
    """
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)
    for z, roi_map in sorted(fatias.items()):
        # Seleciona shift adequado ou zero se fora do alcance
        sx = shifts_x[z] if 0 <= z < len(shifts_x) else 0
        sy = shifts_y[z] if 0 <= z < len(shifts_y) else 0
        fname = os.path.join(output_dir, f"fatia_{z}.txt")
        with open(fname, 'w') as f:
            for pts in roi_map.values():
                for x, y in pts:
                    f.write(f"{x - sx} {y - sy} {z}\n")
        print(f"Saved slice {z} to {fname}")

################################
# 5) Cálculo de centróides por cluster 2D
################################
def compute_centroids_2d(clusters_2d_por_fatia):
    """
    Gera lista de objetos Cluster2D(z, cluster_id, centroid, points)
    para uso no agrupamento 3D.
    """
    from collections import namedtuple
    Cluster2D = namedtuple('Cluster2D', ['z', 'cluster_id', 'centroid', 'points'])
    result = []
    for z, clusters_dict in clusters_2d_por_fatia.items():
        for cid, pts_2d in clusters_dict.items():
            arr = np.array(pts_2d)
            centroid = arr.mean(axis=0)
            result.append(Cluster2D(z, cid, centroid, pts_2d))
    return result

################################
# 6) Conexão de clusters 2D em 3D (formação de cicatrizes)
################################
def min_distance_between_clusters(c1, c2):
    p1 = np.array(c1.points); p2 = np.array(c2.points)
    if p1.size == 0 or p2.size == 0:
        return float('inf')
    # Distância mínima entre qualquer par de pontos
    return np.min(distance.cdist(p1, p2, 'euclidean'))

def connect_2d_clusters_in_3d(clusters_2d_list, base_radius_3d=3.0, max_delta_z=1):
    """
    Constrói um grafo onde arestas unem clusters em fatias adjacentes
    cuja distância (pontos) é menor que um raio dinâmico. Então
    encontra componentes conexas (clusters 3D) via DFS.
    """
    n = len(clusters_2d_list)
    adj = [[] for _ in range(n)]
    # Cria conexões
    for i in range(n):
        for j in range(i+1, n):
            a, b = clusters_2d_list[i], clusters_2d_list[j]
            if a.z == b.z or abs(a.z - b.z) > max_delta_z:
                continue
            d = min_distance_between_clusters(a, b)
            dyn_r = max(base_radius_3d,
                        (len(a.points) + len(b.points)) * 0.05)
            if d < dyn_r:
                adj[i].append(j); adj[j].append(i)
    # Encontra componentes
    visited = [False]*n; comp = [-1]*n; cid=0
    def dfs(u):
        stack=[u]; visited[u]=True; comp[u]=cid
        while stack:
            v=stack.pop()
            for w in adj[v]:
                if not visited[w]: visited[w]=True; comp[w]=cid; stack.append(w)
    for i in range(n):
        if not visited[i]: dfs(i); cid+=1
    # Monta dicionário z->pontos 3D
    clusters_3d = defaultdict(list)
    for i, c2d in enumerate(clusters_2d_list):
        for x, y in c2d.points:
            clusters_3d[comp[i]].append((x, y, c2d.z))
    return dict(clusters_3d)

################################
# 7) Gravação de clusters 3D em .txt
################################
def save_clusters_to_txt(clusters, mat_filename, output_dir="clusters_dbscan"):
    """
    Carrega metadados do .mat para obter resolução e espessura,
    então grava cada cluster de pontos 3D dimensionado.
    """
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)
    ss = data['setstruct']
    slice_thickness = ss['SliceThickness'][0][0][0][0]
    gap = ss['SliceGap'][0][0][0][0]
    resolution_x = ss['ResolutionX'][0][0][0][0]
    resolution_y = ss['ResolutionY'][0][0][0][0]
    os.makedirs(output_dir, exist_ok=True)
    for label, pts in clusters.items():
        fname = os.path.join(output_dir, f"cluster_{label}.txt")
        with open(fname,'w') as f:
            for x, y, z in pts:
                f.write(f"{x*resolution_x} {y*resolution_y} {z*(slice_thickness+gap)}\n")
        print(f"Cluster {label} saved to: {fname}")

################################
# 8) Geração de superfícies (.ply) e STL
################################
def generate_surfaces_and_stl():
    """
    Para cada arquivo cluster_<id>.txt:
      1) chama make_surface.py para gerar .ply
      2) converte .ply em .stl usando PlyToStl
    """
    txts = sorted(glob.glob("./clusters_dbscan/cluster_*.txt"))
    for txt in txts:
        try:
            subprocess.run(f"python3 make_surface.py {txt} --cover-both-ends",
                           shell=True, check=True)
            print(f"Surface for {txt} generated.")
        except subprocess.CalledProcessError as e:
            print(f"Surface error {txt}: {e}")
        base = os.path.splitext(os.path.basename(txt))[0]
        ply = f"./output/plyFiles/{base}.ply"
        stl = f"./output/scarFiles/{base}.stl"
        if os.path.exists(ply):
            try:
                subprocess.run(f"./convertPly2STL/build/PlyToStl {ply} {stl} 1",
                               shell=True, check=True)
                print(f"STL created: {stl}")
            except Exception as e:
                print(f"STL error {ply}: {e}")

################################
# MAIN: execução completa
################################
def main():
    parser = argparse.ArgumentParser(description="Full scar pipeline")
    parser.add_argument('matfile', help='Path to .mat file')
    parser.add_argument('--shiftx', default='endo_shifts_x.txt')
    parser.add_argument('--shifty', default='endo_shifts_y.txt')
    args = parser.parse_args()

    # 1-3: leitura, agrupamento e plot
    entries = readScar(args.matfile)
    fatias = group_by_slice(entries)
    plot_slices(fatias)
    # 4: grava fatias alinhadas
    save_fatias_to_txt(fatias, args.shiftx, args.shifty)

    # 5-7: clustering 2D to 3D e gravação de clusters
    clusters_2d_list = compute_centroids_2d(fatias)
    clusters_3d = connect_2d_clusters_in_3d(clusters_2d_list, base_radius_3d=0.5)
    save_clusters_to_txt(clusters_3d, args.matfile)
    # 8: geração de superfícies e STL
    generate_surfaces_and_stl()

if __name__ == '__main__':
    main()
