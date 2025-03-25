import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

def extrair_rois_com_z(caminho_mat):
    """
    Extrai os ROIs de um arquivo .mat (Segment) com campos X, Y, Z e Name.
    Retorna um dicionário com {nome: {'pontos': array, 'z': valor}}.
    """
    dados = loadmat(caminho_mat)
    rois_extraidos = {}

    # Verifica se 'setstruct' e 'Roi' estão no arquivo
    if 'setstruct' not in dados:
        raise ValueError("Arquivo .mat não contém a chave 'setstruct'.")
    
    setstruct = dados['setstruct']
    if 'Roi' not in setstruct.dtype.names:
        raise ValueError("Estrutura 'setstruct' não contém o campo 'Roi'.")

    rois = setstruct['Roi'][0, 0][0]  # Acessa o vetor de ROIs

    for roi in rois:
        nome = roi['Name'][0]
        z = int(roi['Z'][0][0])
        x = roi['X'].flatten()
        y = roi['Y'].flatten()
        pontos = np.vstack((x, y)).T
        rois_extraidos[nome] = {
            'pontos': pontos,
            'z': z
        }

    return rois_extraidos

def plotar_rois_3d(rois_extraidos):
    """
    Plota os ROIs extraídos em 3D usando Matplotlib.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for nome, info in rois_extraidos.items():
        pontos = info['pontos']
        z = info['z']
        xs, ys = pontos[:, 0], pontos[:, 1]
        zs = np.full_like(xs, z)
        ax.plot(xs, ys, zs, label=nome)

    ax.set_title("ROIs - Visualização 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (fatia)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    caminho_mat = '../segmentacoesPacientes/comRoi/Patient_2.mat'  # Atualize com o caminho real
    rois = extrair_rois_com_z(caminho_mat)
    plotar_rois_3d(rois)
