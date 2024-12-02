import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_mat_data(filename, key):
    """Carrega os dados do arquivo MATLAB com a chave especificada."""
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return data[key]

def ensure_three_dimensions(array):
    """Garante que o array tenha três dimensões, adicionando eixos se necessário."""
    if array.ndim == 2:
        array = array[:, np.newaxis, :]
    elif array.ndim == 1:
        array = array[:, np.newaxis, np.newaxis]
    return array

def calculate_barycenters(X, Y):
    """Calcula os baricentros para cada fatia."""
    barycenters = np.array([
        [np.nanmean(X[:, 0, s]), np.nanmean(Y[:, 0, s])] if not np.isnan(X[:, 0, s]).all() else [np.nan, np.nan]
        for s in range(X.shape[2])
    ])
    return barycenters

def evaluate_alignment(original_filename, aligned_filename):
    """Compara o alinhamento entre as versões original e alinhada."""
    original_data = load_mat_data(original_filename, 'setstruct')
    aligned_data = load_mat_data(aligned_filename, 'SEGsave')

    # Extrair coordenadas das estruturas e garantir 3D
    endoX_orig, endoY_orig = ensure_three_dimensions(original_data.EndoX), ensure_three_dimensions(original_data.EndoY)
    epiX_orig, epiY_orig = ensure_three_dimensions(original_data.EpiX), ensure_three_dimensions(original_data.EpiY)

    endoX_aligned, endoY_aligned = ensure_three_dimensions(aligned_data.EndoX), ensure_three_dimensions(aligned_data.EndoY)
    epiX_aligned, epiY_aligned = ensure_three_dimensions(aligned_data.EpiX), ensure_three_dimensions(aligned_data.EpiY)

    # Calcular os baricentros
    barycenters_endo_orig = calculate_barycenters(endoX_orig, endoY_orig)
    barycenters_epi_orig = calculate_barycenters(epiX_orig, epiY_orig)

    barycenters_endo_aligned = calculate_barycenters(endoX_aligned, endoY_aligned)
    barycenters_epi_aligned = calculate_barycenters(epiX_aligned, epiY_aligned)

    # Calcular desvios padrão dos baricentros para avaliar o alinhamento
    std_endo_orig = np.nanstd(barycenters_endo_orig, axis=0)
    std_epi_orig = np.nanstd(barycenters_epi_orig, axis=0)

    std_endo_aligned = np.nanstd(barycenters_endo_aligned, axis=0)
    std_epi_aligned = np.nanstd(barycenters_epi_aligned, axis=0)

    # Exibir os resultados
    print("Desvios padrão dos baricentros (python):")
    print(f"Endo: X = {std_endo_orig[0]:.4f}, Y = {std_endo_orig[1]:.4f}")
    print(f"Epi: X = {std_epi_orig[0]:.4f}, Y = {std_epi_orig[1]:.4f}")

    print("\nDesvios padrão dos baricentros (matlab):")
    print(f"Endo: X = {std_endo_aligned[0]:.4f}, Y = {std_endo_aligned[1]:.4f}")
    print(f"Epi: X = {std_epi_aligned[0]:.4f}, Y = {std_epi_aligned[1]:.4f}")

    # Plotar comparação visual dos baricentros
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Baricentros - python")
    plt.plot(barycenters_endo_orig[:, 0], barycenters_endo_orig[:, 1], 'bo-', label='Endo')
    plt.plot(barycenters_epi_orig[:, 0], barycenters_epi_orig[:, 1], 'ro-', label='Epi')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Baricentros - matlab")
    plt.plot(barycenters_endo_aligned[:, 0], barycenters_endo_aligned[:, 1], 'bo-', label='Endo')
    plt.plot(barycenters_epi_aligned[:, 0], barycenters_epi_aligned[:, 1], 'ro-', label='Epi')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Exemplo de uso
evaluate_alignment('analise.mat', 'Patient_1_aligned.mat')
