import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_comparison(original_filename, processed_filename, slice_index, plot_original=True, min_valid_ratio=0.1):
    """
    Plota uma comparação entre as estruturas originais e processadas para uma fatia específica.
    
    Parameters:
    - original_filename: Caminho para o arquivo MATLAB original.
    - processed_filename: Caminho para o arquivo MATLAB processado.
    - slice_index: Índice da fatia a ser plotada.
    - plot_original: Booleano para decidir se os dados originais serão plotados.
    - min_valid_ratio: Fração mínima de pontos válidos para considerar a fatia.
    """
    # Carregar os dados processados
    processed_data = loadmat(processed_filename)['setstruct'][0][0]

    def extract_coordinates(data):
        endoX, endoY = data['EndoX'], data['EndoY']
        RVEndoX, RVEndoY = data['RVEndoX'], data['RVEndoY']
        RVEpiX, RVEpiY = data['RVEpiX'], data['RVEpiY']
        return (endoX, endoY, RVEndoX, RVEndoY, RVEpiX, RVEpiY)

    # Extrair coordenadas do processado
    proc_endoX, proc_endoY, proc_rvendoX, proc_rvendoY, proc_rvepiX, proc_rvepiY = extract_coordinates(processed_data)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    def plot_structure(ax, X, Y, color, label):
        ax.scatter(
            X.flatten(), Y.flatten(),
            color=color, s=15, alpha=0.8,
            edgecolor='black', linewidth=0.5,
            label=label
        )

    def plot_valid_points(ax, X, Y, color, label, s):
        """Plota apenas pontos válidos (não NaN) para uma fatia específica."""
        if X.ndim == 3:
            valid = ~np.isnan(X[:, 0, s]) & ~np.isnan(Y[:, 0, s])
            plot_structure(ax, X[valid, 0, s], Y[valid, 0, s], color, label)
        elif X.ndim == 2:
            valid = ~np.isnan(X[:, s]) & ~np.isnan(Y[:, s])
            plot_structure(ax, X[valid, s], Y[valid, s], color, label)

    # Função para validar uma fatia com base na proporção mínima de pontos válidos
    def valid_slice(data, s, min_valid_ratio):
        valid_points = np.sum(~np.isnan(data[:, 0, s]))
        return valid_points >= min_valid_ratio * data.shape[0]

    # Plotar as estruturas originais (se habilitado)
    if plot_original:
        original_data = loadmat(original_filename)['SEGsave'][0][0]
        orig_endoX, orig_endoY, orig_rvendoX, orig_rvendoY, orig_rvepiX, orig_rvepiY = extract_coordinates(original_data)
        
        if slice_index < orig_endoX.shape[2] and valid_slice(orig_endoX, slice_index, min_valid_ratio):
            plot_valid_points(ax, orig_endoX, orig_endoY, '#1f77b4', 'Original - Endo', slice_index)
            plot_valid_points(ax, orig_rvendoX, orig_rvendoY, '#2ca02c', 'Original - RVEndo', slice_index)
            plot_valid_points(ax, orig_rvepiX, orig_rvepiY, '#d62728', 'Original - RVEpi', slice_index)
        else:
            print(f"Fatia {slice_index} inválida ou fora do alcance nos dados originais.")
    else:
        if slice_index >= proc_endoX.shape[2]:
            print(f"Fatia {slice_index} inválida ou fora do alcance nos dados processados.")
            return

    # Plotar as estruturas processadas para a fatia especificada
    if valid_slice(proc_endoX, slice_index, min_valid_ratio):
        plot_valid_points(ax, proc_endoX, proc_endoY, '#17becf', 'Processado - Endo', slice_index)
        plot_valid_points(ax, proc_rvendoX, proc_rvendoY, '#98df8a', 'Processado - RVEndo', slice_index)
        plot_valid_points(ax, proc_rvepiX, proc_rvepiY, '#ff9896', 'Processado - RVEpi', slice_index)
    else:
        print(f"Fatia {slice_index} inválida ou fora do alcance nos dados processados.")

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title(f'Comparação das Estruturas - Fatia {slice_index}')
    ax.legend(loc='upper right')

    plt.show()

# Exemplo de uso
plot_comparison('Patient_1_aligned.mat', 'analise.mat', slice_index=12, plot_original=True)
