import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_slices_with_time(coordX_dict, coordY_dict, num_slices):
    """
    Plota as coordenadas X e Y ao longo do tempo para cada estrutura.
    """
    for key in coordX_dict.keys():
        plt.figure(figsize=(10, 5))
        plt.title(f'{key} - Movimento ao Longo do Tempo')
        plt.xlabel('Fatia')
        plt.ylabel('Coordenada')

        # Plot das coordenadas X e Y ao longo do tempo para a estrutura
        x_mean = [np.nanmean(coordX_dict[key][:, 0, s]) for s in range(num_slices)]
        y_mean = [np.nanmean(coordY_dict[key][:, 0, s]) for s in range(num_slices)]

        plt.plot(range(num_slices), x_mean, '-o', label=f'{key} - X', alpha=0.7)
        plt.plot(range(num_slices), y_mean, '-o', label=f'{key} - Y', alpha=0.7)

        plt.legend()
        plt.grid(True)
        plt.show()

# Carrega o arquivo alinhado .mat
data = loadmat('analise.mat')

# Acessa a estrutura setstruct
setstruct = data['setstruct'][0][0]

# Extrai as coordenadas de cada estrutura
coordX_dict = {
    'Endo': setstruct['EndoX'],
    'RVEndo': setstruct['RVEndoX'],
    'RVEpi': setstruct['RVEpiX']
}

coordY_dict = {
    'Endo': setstruct['EndoY'],
    'RVEndo': setstruct['RVEndoY'],
    'RVEpi': setstruct['RVEpiY']
}

# Define o número de fatias
num_slices = coordX_dict['Endo'].shape[2]

# Chama a função de plotagem
print("Plotando movimento das estruturas ao longo do tempo...")
plot_slices_with_time(coordX_dict, coordY_dict, num_slices)

print("Diagnóstico concluído.")
