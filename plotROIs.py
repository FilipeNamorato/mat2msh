import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

def readScar(mat_filename, plot_3d=True):
    """
    Lê os ROIs do arquivo .mat e plota os pontos em 3D.

    Parâmetros:
      - mat_filename: Caminho do arquivo .mat
      - plot_3d: Se True, plota os pontos extraídos.

    Retorna:
      - fatias: dict { z_val: [(x, y), (x, y), ...] }
      - pontos_3d: np.array (N, 3) com os pontos extraídos.
    """
    print(f"Lendo arquivo: {mat_filename}")
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
                    print(f"Error: Slice {z_val} from ROI {idx+1} is empty.")
                    continue

                if z_val not in fatias:
                    fatias[z_val] = []
                
                # Add points (x, y) to this slice
                fatias[z_val].extend(zip(x_arr, y_arr))

                print(f"Added ROI {idx+1}, Slice {z_val}, Points={len(x_arr)}")

        except Exception as e:
            print(f"Error while processing ROI {idx+1}: {e}")

    # Cria um array unificado para debug
    pontos_3d = np.array([[x, y, z] for z, coords in fatias.items() for (x, y) in coords])

    if plot_3d:
        plot_rois_3d(pontos_3d)

    return fatias, pontos_3d

def plot_rois_3d(pontos_3d):
    """
    Plota os ROIs em 3D.
    
    Parâmetros:
      - pontos_3d: np.array (N, 3) contendo (x, y, z) das fibroses
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = pontos_3d[:, 0]
    y = pontos_3d[:, 1]
    z = pontos_3d[:, 2]

    ax.scatter(x, y, z, c='blue', s=5, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Distribuição 3D dos ROIs")

    plt.show()


# Executa a leitura e plota os ROIs em 3D
mat_filename = "Patient_1_new.mat"
fatias_data, pontos_3d = readScar(mat_filename, plot_3d=True)
