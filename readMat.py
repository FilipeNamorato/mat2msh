import numpy as np
from scipy.stats import linregress
from scipy.io import loadmat, savemat

def read_mat(mat_filename):
    # Carrega o arquivo .mat e extrai o campo 'setstruct'
    data = loadmat(mat_filename)
    mat = data['setstruct'][0][0]

    # Verifica se o RV está presente
    RVyes = 'RVEndoX' in mat.dtype.names

    # Configurações básicas
    N = mat['EndoX'].shape[2]  # Número de fatias (frames)
    SliceThickness = mat['SliceThickness'].item()
    zEndo = np.arange(1, N + 1) * SliceThickness
    zEpi = np.arange(1, mat['EpiX'].shape[2] + 1) * SliceThickness

    if RVyes:
        zRV = np.arange(1, mat['RVEndoX'].shape[2] + 1) * SliceThickness

    def align_coordinates(coordX, coordY, z, label):
        # Fazer uma cópia dos dados para evitar modificações acidentais
        coordX = np.copy(coordX)
        coordY = np.copy(coordY)

        # Calcular a média ignorando NaNs, mas pular fatias vazias
        X_avgs = np.nanmean(coordX, axis=0).squeeze()
        Y_avgs = np.nanmean(coordY, axis=0).squeeze()

        # Verificar se os dados têm a estrutura correta
        if X_avgs.ndim == 1:
            X_avgs = X_avgs[:, np.newaxis]
            Y_avgs = Y_avgs[:, np.newaxis]

        for n in range(X_avgs.shape[1]):
            # Verificar se a fatia é vazia (toda NaN)
            if np.isnan(X_avgs[:, n]).all() or np.isnan(Y_avgs[:, n]).all():
                print(f"Fatia {n} está vazia, pulando...")
                continue  # Pular esta fatia

            try:
                # Identificar índices válidos para regressão
                valid_idx = ~np.isnan(X_avgs[:, n]) & ~np.isnan(Y_avgs[:, n])

                if valid_idx.sum() > 1:  # Precisamos de pelo menos 2 pontos
                    slopeX, interceptX, _, _, _ = linregress(z[valid_idx], X_avgs[valid_idx, n])
                    slopeY, interceptY, _, _, _ = linregress(z[valid_idx], Y_avgs[valid_idx, n])

                    estX = interceptX + slopeX * z
                    estY = interceptY + slopeY * z

                    for s in range(z.size):
                        if not np.isnan(X_avgs[s, n]) and not np.isnan(Y_avgs[s, n]):
                            shiftX = X_avgs[s, n] - estX[s]
                            shiftY = Y_avgs[s, n] - estY[s]

                            # Aplicar o deslocamento apenas em valores válidos
                            coordX[:, n, s] -= shiftX
                            coordY[:, n, s] -= shiftY
            except ValueError as e:
                print(f"Erro na regressão para a fatia {n}: {e}. Pulando...")

        # Atualizar o dicionário com os dados alterados
        mat_shift[label + 'X'] = coordX
        mat_shift[label + 'Y'] = coordY


    # Inicializar o dicionário para armazenar mudanças
    mat_shift = {}

    # Alinhar endocárdio e epicárdio
    align_coordinates(mat['EndoX'], mat['EndoY'], zEndo, 'Endo')
    align_coordinates(mat['EpiX'], mat['EpiY'], zEpi, 'Epi')

    # Alinhar o RV se presente
    if RVyes:
        align_coordinates(mat['RVEndoX'], mat['RVEndoY'], zRV, 'RVEndo')
        align_coordinates(mat['RVEpiX'], mat['RVEpiY'], zRV, 'RVEpi')

    return mat_shift
