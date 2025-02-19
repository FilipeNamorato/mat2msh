import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_heart_slices_and_clusters(cluster_dir="clusters_dbscan", heart_slices_dir="heart_slices"):
    """Plota os clusters e as fatias do coração (LV endo, RV, etc.) a partir de arquivos .txt."""
    
    # Encontrar arquivos .txt de clusters e fatias cardíacas
    cluster_files = glob.glob(os.path.join(cluster_dir, "*.txt"))
    heart_slice_files = glob.glob(os.path.join(heart_slices_dir, "*.txt"))
    
    if not cluster_files and not heart_slice_files:
        print("Nenhum arquivo encontrado para clusters ou fatias cardíacas.")
        return
    
    print(f"Clusters encontrados: {cluster_files}")
    print(f"Fatias cardíacas encontradas: {heart_slice_files}")
    
    # Criar figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Paletas de cores diferentes para diferenciar clusters e fatias
    cluster_colors = plt.get_cmap("tab10")
    heart_colors = plt.get_cmap("Set1")
    
    # 1) Plotar clusters
    for i, cluster_file in enumerate(cluster_files):
        data = np.loadtxt(cluster_file, skiprows=1)  # Ignorar cabeçalho (se houver)
        if data.shape[0] == 0:
            print(f"Aviso: {cluster_file} está vazio. Pulando...")
            continue

        # Multiplicar X, Y, Z se precisar
        xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]

        # Se quiser inverter a fatia no cluster também, faça o mesmo cálculo do minZ/maxZ
        minZ, maxZ = np.min(zs), np.max(zs)
        zs_invertidos = (maxZ + minZ) - zs  # inverte

        color = cluster_colors(i % 10)
        ax.scatter(xs, ys, zs_invertidos, c=[color], label=f"", s=10)
    
    # 2) Plotar fatias do coração
    for i, slice_file in enumerate(heart_slice_files):
        data = np.loadtxt(slice_file, skiprows=1)
        if data.shape[0] == 0:
            print(f"Aviso: {slice_file} está vazio. Pulando...")
            continue
        
        xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]

        color = heart_colors(i % 9)
        ax.scatter(xs, ys, zs, c=[color], label="", marker='o', s=15)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Clusters e Fatias Cardíacas 3D - Z Invertido")
    
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    plot_heart_slices_and_clusters("./clusters_dbscan/", "./output/20250219/")
