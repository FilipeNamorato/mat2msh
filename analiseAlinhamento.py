import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# Carrega os arquivos .mat
original_data = sio.loadmat('Patient_1.mat')
modified_data = sio.loadmat('Teste2.mat')

# Extrai as estruturas relevantes
original_setstruct = original_data['setstruct'][0][0]
modified_setstruct = modified_data['setstruct'][0][0]

# Função para calcular o baricentro ao longo das fatias
def calculate_barycenter_along_slices(coordX, coordY):
    barycenter_X = np.nanmean(coordX, axis=0)  # Média ao longo dos pontos
    barycenter_Y = np.nanmean(coordY, axis=0)
    return barycenter_X, barycenter_Y

# Função para exibir informações úteis no console
def print_alignment_info(field, original, modified):
    orig_bary_X, orig_bary_Y = calculate_barycenter_along_slices(
        original[field], original[field.replace('X', 'Y')]
    )
    mod_bary_X, mod_bary_Y = calculate_barycenter_along_slices(
        modified[field], modified[field.replace('X', 'Y')]
    )

    # Calcula o deslocamento médio aplicado em cada fatia
    displacement_X = np.nanmean(mod_bary_X - orig_bary_X)
    displacement_Y = np.nanmean(mod_bary_Y - orig_bary_Y)

    # Calcula o desvio padrão dos baricentros (antes e depois)
    std_orig_X = np.nanstd(orig_bary_X)
    std_mod_X = np.nanstd(mod_bary_X)

    print(f"\n[Field: {field}]")
    print(f"  Deslocamento Médio em X: {displacement_X:.2f}")
    print(f"  Deslocamento Médio em Y: {displacement_Y:.2f}")
    print(f"  Desvio Padrão Antes (X): {std_orig_X:.2f}")
    print(f"  Desvio Padrão Depois (X): {std_mod_X:.2f}")

# Função para plotar os baricentros
def plot_barycenter_comparison(field, original, modified):
    orig_bary_X, _ = calculate_barycenter_along_slices(
        original[field], original[field.replace('X', 'Y')]
    )
    mod_bary_X, _ = calculate_barycenter_along_slices(
        modified[field], modified[field.replace('X', 'Y')]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(orig_bary_X, label='Original Barycenter X', marker='o')
    plt.plot(mod_bary_X, label='Aligned Barycenter X', marker='x')
    plt.title(f'Barycenter Comparison - {field}')
    plt.xlabel('Slice Index')
    plt.ylabel('Barycenter X Coordinate')
    plt.legend()
    plt.show()

# Avaliação das estruturas
fields_to_compare = ['EndoX', 'RVEndoX', 'RVEpiX']

for field in fields_to_compare:
    print_alignment_info(field, original_setstruct, modified_setstruct)
    plot_barycenter_comparison(field, original_setstruct, modified_setstruct)
