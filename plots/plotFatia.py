import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_data(filename):
    """
    Carrega os dados do arquivo .mat e identifica a estrutura ('setstruct' ou 'SEGsave').
    """
    mat_data = loadmat(filename)
    if 'setstruct' in mat_data:
        structure_key = 'setstruct'
        data = mat_data[structure_key][0][0]
    elif 'SEGsave' in mat_data:
        structure_key = 'SEGsave'
        data = mat_data[structure_key][0][0]
    else:
        raise ValueError("Arquivo .mat não contém 'setstruct' ou 'SEGsave'.")
    return data, structure_key

def extract_coordinates(data, structure_key):
    """
    Extrai as coordenadas das estruturas 'setstruct' ou 'SEGsave' e aplica os fatores de escala.
    """
    if structure_key == 'setstruct':
        # Extrair resoluções
        resolution_x = data['ResolutionX'][0, 0]
        resolution_y = data['ResolutionY'][0, 0]
        
        # Extrair e ajustar coordenadas
        endoX, endoY = data['EndoX'] * resolution_x, data['EndoY'] * resolution_y
        RVEndoX, RVEndoY = data['RVEndoX'] * resolution_x, data['RVEndoY'] * resolution_y
        RVEpiX, RVEpiY = data['RVEpiX'] * resolution_x, data['RVEpiY'] * resolution_y
    elif structure_key == 'SEGsave':
        endoX = data['EndoXnew'][0, 0]
        endoY = data['EndoYnew'][0, 0]
        RVEndoX = data['RVEndoXnew'][0, 0]
        RVEndoY = data['RVEndoYnew'][0, 0]
        RVEpiX = data['RVEpiXnew'][0, 0]
        RVEpiY = data['RVEpiYnew'][0, 0]
    else:
        raise ValueError("Estrutura desconhecida no arquivo .mat.")
    return (endoX, endoY, RVEndoX, RVEndoY, RVEpiX, RVEpiY)


def plot_structure(ax, X, Y, Z, color, label):
    """
    Plota os pontos fornecidos nas coordenadas X, Y e Z.
    """
    ax.scatter(
        X.flatten(), Y.flatten(), Z, 
        color=color, s=15, alpha=0.8,
        edgecolor='black', linewidth=0.5,
        label=label
    )

def plot_valid_points(ax, X, Y, slice_index, color, label):
    """
    Plota apenas pontos válidos (não NaN) para uma fatia específica.
    """
    print(f"Debug: Dimensões de X: {X.shape}, Dimensões de Y: {Y.shape}")
    if X.ndim == 3:  # Dados tridimensionais
        valid = ~np.isnan(X[:, 0, slice_index]) & ~np.isnan(Y[:, 0, slice_index])
        if np.sum(valid) > 0:
            plot_structure(ax, X[valid, 0, slice_index], Y[valid, 0, slice_index],
                           np.full(np.sum(valid), slice_index), color, label)
    elif X.ndim == 2:  # Dados bidimensionais
        valid = ~np.isnan(X[:, slice_index]) & ~np.isnan(Y[:, slice_index])
        if np.sum(valid) > 0:
            plot_structure(ax, X[valid, slice_index], Y[valid, slice_index],
                           np.full(np.sum(valid), slice_index), color, label)
    elif X.ndim == 1:  # Dados unidimensionais
        valid = ~np.isnan(X) & ~np.isnan(Y)
        if np.sum(valid) > 0:
            plot_structure(ax, X[valid], Y[valid], np.full(np.sum(valid), slice_index), color, label)
    else:
        raise ValueError(f"Dimensão inesperada dos dados de coordenadas: X={X.shape}, Y={Y.shape}")

def plot_comparison(original_filename, processed_filename, plot_original=True):
    """
    Plota uma comparação entre as estruturas originais e processadas.
    """
    # Carregar os dados originais
    original_data, original_key = load_data(original_filename)
    orig_endoX, orig_endoY, orig_rvendoX, orig_rvendoY, orig_rvepiX, orig_rvepiY = extract_coordinates(original_data, original_key)

    # Carregar os dados processados
    processed_data, processed_key = load_data(processed_filename)
    proc_endoX, proc_endoY, proc_rvendoX, proc_rvendoY, proc_rvepiX, proc_rvepiY = extract_coordinates(processed_data, processed_key)

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')

    def consider_fatia(fatia_idx):
        """
        Verifica se pelo menos um conjunto de dados da fatia é válido.
        """
        def is_valid(array):
            if array.ndim == 3:
                return np.any(~np.isnan(array[:, :, fatia_idx]))
            elif array.ndim == 2:
                return np.any(~np.isnan(array[:, fatia_idx]))
            elif array.ndim == 1:
                return np.any(~np.isnan(array))
            else:
                return False

        return (
            is_valid(orig_endoX) or
            is_valid(orig_rvendoX) or
            is_valid(orig_rvepiX)
        )

    num_slices = min(orig_endoX.shape[-1], proc_endoX.shape[-1])

    # Plotar as estruturas originais e processadas
    for slice_index in range(num_slices):
        if consider_fatia(slice_index):
            if plot_original:
                plot_valid_points(ax, orig_endoX, orig_endoY, slice_index, '#1f77b4', 'Original - Endo')
                plot_valid_points(ax, orig_rvendoX, orig_rvendoY, slice_index, '#2ca02c', 'Original - RVEndo')
                plot_valid_points(ax, orig_rvepiX, orig_rvepiY, slice_index, '#d62728', 'Original - RVEpi')

            plot_valid_points(ax, proc_endoX, proc_endoY, slice_index, '#17becf', 'Processado - Endo')
            plot_valid_points(ax, proc_rvendoX, proc_rvendoY, slice_index, '#98df8a', 'Processado - RVEndo')
            plot_valid_points(ax, proc_rvepiX, proc_rvepiY, slice_index, '#ff9896', 'Processado - RVEpi')

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Fatia (Eixo Z)')
    ax.set_title('Comparação das Estruturas - Original vs Processado')
    ax.legend(loc='upper right')

    plt.show()

# Exemplo de uso
#plot_comparison('Patient_1_aligned.mat', 'analise_alinhada.mat', plot_original=False)
plot_comparison('analise_alinhada.mat', 'analise_alinhada.mat', plot_original=False)