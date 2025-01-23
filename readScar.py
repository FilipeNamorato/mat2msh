from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def readScar():
    mat_filename = "Patient_1_new.mat"
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)  # Carrega o arquivo
    setstruct = data['setstruct']
    
    # Acessando o campo 'Roi' e os subcampos dinamicamente
    rois = setstruct[0][0]['Roi']

    records = []
    for idx, roi in enumerate(rois):
        print(f"Processando ROI {idx+1}...")
        try:
            x_coords = roi['X'] if 'X' in roi.dtype.names else None
            y_coords = roi['Y'] if 'Y' in roi.dtype.names else None
            z_values = roi['Z'] if 'Z' in roi.dtype.names else None
            roi_name = roi['Name'][0] if 'Name' in roi.dtype.names else f"ROI-{idx+1}"
            roi_name = str(roi_name)

            # Verificar se os tamanhos de X, Y e Z são consistentes
            if x_coords is None or y_coords is None or z_values is None:
                print(f"Erro: Dados incompletos para ROI {roi_name}")
                continue

            if len(x_coords) != len(y_coords) or len(x_coords) != len(z_values):
                print(f"Erro: Tamanhos inconsistentes para ROI {roi_name}")
                continue

            # Processar cada fatia de Z com seus X e Y correspondentes
            for i in range(len(z_values)):
                x = x_coords[i].flatten()
                y = y_coords[i].flatten()
                z = float(z_values[i][0][0])  # Extrair o valor escalar de Z
                records.append({
                    'X': x,
                    'Y': y,
                    'Z': np.full_like(x, z),
                    'Name': f"{roi_name} - Fatia {i+1}"
                })
                print(f"Adicionado ROI {roi_name}, Fatia {i+1}: Z={z}")
        except Exception as e:
            print(f"Erro ao processar ROI {idx+1}: {e}")

    return records

def plot_all_fatias(roi_records):
    """
    Plota todos os pontos das fatias em 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, roi in enumerate(roi_records):
        try:
            x_coords = roi['X']
            y_coords = roi['Y']
            z_coords = roi['Z']
            roi_name = roi['Name']

            # Logs de depuração
            print(f"Plotando ROI {idx+1}: X={x_coords.shape}, Y={y_coords.shape}, Z={z_coords.shape}")

            # Plota os pontos da fatia
            ax.scatter(x_coords, y_coords, z_coords, label=roi_name, s=10)
        except Exception as e:
            print(f"Erro ao plotar ROI {idx+1}: {e}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Visualização 3D de Todas as Fatias (ROI)")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    roi_records = readScar()
    plot_all_fatias(roi_records)
