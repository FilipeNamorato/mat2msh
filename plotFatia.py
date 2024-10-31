import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_comparison(original_filename, processed_filename, plot_original=True, min_valid_ratio=0.1):
    """
    Plota uma comparação entre as estruturas originais e processadas.
    
    Parameters:
    - original_filename: Caminho para o arquivo MATLAB original.
    - processed_filename: Caminho para o arquivo MATLAB processado.
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

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')

    def plot_structure(ax, X, Y, Z, color, label):
        ax.scatter(
            X.flatten(), Y.flatten(), Z, 
            color=color, s=15, alpha=0.8,
            edgecolor='black', linewidth=0.5,
            label=label
        )

    num_slices = proc_endoX.shape[2]

    def plot_valid_points(ax, X, Y, Z, color, label, s):
        """Plota apenas pontos válidos (não NaN) para uma fatia específica."""
        valid = ~np.isnan(X[:, 0, s]) & ~np.isnan(Y[:, 0, s])
        if np.sum(valid) > 0:  # Se houver pontos válidos
            plot_structure(ax, X[valid, 0, s], Y[valid, 0, s], np.full(np.sum(valid), s), color, label)

    # Função para validar uma fatia com base na proporção mínima de pontos válidos
    def valid_slice(data, s, min_valid_ratio):
        valid_points = np.sum(~np.isnan(data[:, 0, s]))
        return valid_points >= min_valid_ratio * data.shape[0]

    # Plotar as estruturas originais (se habilitado)
    if plot_original:
        original_data = loadmat(original_filename)['setstruct'][0][0]
        orig_endoX, orig_endoY, orig_rvendoX, orig_rvendoY, orig_rvepiX, orig_rvepiY = extract_coordinates(original_data)

        for s in range(num_slices):
            plot_valid_points(ax, orig_endoX, orig_endoY, s, '#1f77b4', 'Original - Endo' if s == 0 else "", s)
            plot_valid_points(ax, orig_rvendoX, orig_rvendoY, s, '#2ca02c', 'Original - RVEndo' if s == 0 else "", s)
            plot_valid_points(ax, orig_rvepiX, orig_rvepiY, s, '#d62728', 'Original - RVEpi' if s == 0 else "", s)

    # Plotar as estruturas processadas
    for s in range(num_slices):
        plot_valid_points(ax, proc_endoX, proc_endoY, s, '#17becf', 'Processado - Endo' if s == 0 else "", s)
        plot_valid_points(ax, proc_rvendoX, proc_rvendoY, s, '#98df8a', 'Processado - RVEndo' if s == 0 else "", s)
        plot_valid_points(ax, proc_rvepiX, proc_rvepiY, s, '#ff9896', 'Processado - RVEpi' if s == 0 else "", s)

    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Fatia (Eixo Z)')
    ax.set_title('Comparação das Estruturas - Original vs Processado')
    ax.legend(loc='upper right')

    plt.show()

# Exemplo de uso
plot_comparison('Patient_1.mat', 'analise.mat', plot_original=False)
