import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_clusters_from_txt(input_dir="clusters_dbscan"):
    """Lê arquivos .txt contendo coordenadas X, Y, Z e plota cada cluster em 3D."""
    pattern = os.path.join(input_dir, "fatia_7.txt")
    txt_files = glob.glob(pattern)

    if not txt_files:
        print(f"Nenhum arquivo encontrado em {pattern}")
        return

    print(f"Arquivos encontrados: {txt_files}")

    # Criar figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Paleta de cores
    colors = plt.colormaps.get_cmap("tab10")  # Correção aqui

    for i, txt_file in enumerate(txt_files):
        data = []
        with open(txt_file, "r") as file:
            file.readline()  # Ignorar cabeçalho
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    x, y, z = map(float, line.split(" "))
                    data.append((x, y, z))
                except ValueError:
                    print(f"Aviso: linha inválida ignorada em {txt_file} -> {line}")

        data = np.array(data)
        if data.shape[0] == 0:
            print(f"Aviso: {txt_file} está vazio. Pulando...")
            continue

        # Separar colunas X, Y, Z
        xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]

        # Escolher cor da paleta (ciclo se houver mais de 10 clusters)
        color = colors(i % 10)

        # Plotar em 3D com cor distinta
        ax.scatter(xs, ys, zs, c=[color], label=os.path.basename(txt_file), s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Visualização 3D dos Clusters")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_clusters_from_txt("./fatias_txt")
