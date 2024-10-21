import numpy as np
from scipy.io import loadmat, savemat

def read_mat(mat_filename):
    # Carrega o arquivo .mat e extrai o campo 'setstruct'
    data = loadmat(mat_filename)
    mat = data['setstruct'][0][0]

    # Função para calcular o baricentro ignorando a série temporal
    def calculate_barycenter_ignoring_time(coordX, coordY):
        # Calcula a média ao longo dos frames (dimensão 1)
        X_avg = np.nanmean(coordX, axis=1, keepdims=True)
        Y_avg = np.nanmean(coordY, axis=1, keepdims=True)
        return X_avg.squeeze(), Y_avg.squeeze()

    def shift_structure(coordX, coordY, barycenter_X, barycenter_Y, global_barycenter_X, global_barycenter_Y):
        num_slices = coordX.shape[2]  # Número de fatias

        print(f"Shifting structure: {coordX.shape}, {barycenter_X.shape}, {global_barycenter_X.shape}")

        for s in range(num_slices):
            # Garantir que estamos lidando com valores escalares em vez de arrays
            if not np.isnan(barycenter_X[0, s]) and not np.isnan(barycenter_Y[0, s]):
                # Extrair valores escalares para o deslocamento
                shiftX = float(global_barycenter_X[0, s]) - float(barycenter_X[0, s])
                shiftY = float(global_barycenter_Y[0, s]) - float(barycenter_Y[0, s])

                # Aplicar o deslocamento para todos os pontos na fatia
                coordX[:, 0, s] += shiftX  # Usando frame 0 (ignorar série temporal)
                coordY[:, 0, s] += shiftY

    # Extrai as coordenadas das diferentes estruturas
    endoX, endoY = mat['EndoX'], mat['EndoY']
    RVEndoX, RVEndoY = mat['RVEndoX'], mat['RVEndoY']
    RVEpiX, RVEpiY = mat['RVEpiX'], mat['RVEpiY']

    # Calcula os baricentros ignorando a série temporal
    endo_bary_X, endo_bary_Y = calculate_barycenter_ignoring_time(endoX, endoY)
    RV_bary_X, RV_bary_Y = calculate_barycenter_ignoring_time(RVEndoX, RVEndoY)
    epi_bary_X, epi_bary_Y = calculate_barycenter_ignoring_time(RVEpiX, RVEpiY)

    # Calcula os baricentros globais para todas as fatias
    global_barycenter_X = np.nanmean([endo_bary_X, RV_bary_X, epi_bary_X], axis=0)
    global_barycenter_Y = np.nanmean([endo_bary_Y, RV_bary_Y, epi_bary_Y], axis=0)

    # Aplica o deslocamento para cada estrutura, ignorando a série temporal
    shift_structure(endoX, endoY, endo_bary_X, endo_bary_Y, global_barycenter_X, global_barycenter_Y)
    shift_structure(RVEndoX, RVEndoY, RV_bary_X, RV_bary_Y, global_barycenter_X, global_barycenter_Y)
    shift_structure(RVEpiX, RVEpiY, epi_bary_X, epi_bary_Y, global_barycenter_X, global_barycenter_Y)

    # Atualiza as coordenadas no dicionário
    mat['EndoX'], mat['EndoY'] = endoX, endoY
    mat['RVEndoX'], mat['RVEndoY'] = RVEndoX, RVEndoY
    mat['RVEpiX'], mat['RVEpiY'] = RVEpiX, RVEpiY

    return mat
