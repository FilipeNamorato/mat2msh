import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_clusters_from_txt(input_dir="clusters_dbscan"):
    """
    Lê todos os arquivos .txt no diretório especificado, cada um contendo:
        X,Y,Z
        10.0,20.0,5.0
        15.0,25.0,5.0
        ...
    Ignora a primeira linha (cabeçalho).
    Plota cada arquivo em 3D com uma cor distinta.
    """
    # Encontrar todos os arquivos .txt no diretório que comecem com 'cluster_'
    pattern = os.path.join(input_dir, "cluster_*.txt")
    txt_files = glob.glob(pattern)

    if not txt_files:
        print(f"Nenhum arquivo encontrado em {pattern}")
        return

    print(f"Arquivos encontrados: {txt_files}")

    # Criar figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Paleta de cores (se houver muitos arquivos, repetirá)
    colors = plt.cm.get_cmap("tab10", len(txt_files))

    for i, txt_file in enumerate(txt_files):
        # Ler os dados do arquivo
        data = []
        with open(txt_file, "r") as file:
            # pular a primeira linha de cabeçalho
            header = file.readline()
            for line in file:
                line = line.strip()
                if not line:
                    continue
                x_str, y_str, z_str = line.split(",")
                x_val = float(x_str)
                y_val = float(y_str)
                z_val = float(z_str)
                data.append((x_val, y_val, z_val))

        data = np.array(data)
        if data.shape[0] == 0:
            print(f"Aviso: {txt_file} está vazio. Pulando...")
            continue

        # Separar colunas X, Y, Z
        xs = data[:, 0]
        ys = data[:, 1]
        zs = data[:, 2]

        # Plotar em 3D com cor distinta
        ax.scatter(xs, ys, zs, c=[colors(i)], label=os.path.basename(txt_file), s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Visualização 3D dos Clusters (ou Arquivos) em TXT")

    # Mostrar legenda (pode ficar confusa se houver muitos arquivos)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Exemplo de uso, assumindo que os arquivos estão em "clusters_dbscan"
    plot_clusters_from_txt("clusters_dbscan")
