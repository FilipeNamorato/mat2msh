import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LinearRegression

def read_mat(mat_filename):
    print(f"Lendo o arquivo: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']
    print("Estrutura de dados carregada com sucesso.")

    def get_coordinates(field):
        print(f"Extraindo coordenadas para: {field}")
        coords = getattr(setstruct, field, None)
        if coords is None:
            raise ValueError(f"Campo {field} não encontrado.")
        if len(coords.shape) == 2:
            coords = coords[:, np.newaxis, :]
        print(f"Coordenadas para {field} extraídas com shape: {coords.shape}")
        return coords

    # Extrair as coordenadas das estruturas
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    epiX = get_coordinates('EpiX')
    epiY = get_coordinates('EpiY')

    def calculate_barycenters(X, Y):
        """Calcula os baricentros para cada fatia."""
        print("Calculando baricentros...")
        barycenters = np.array([
            [np.nanmean(X[:, 0, s]), np.nanmean(Y[:, 0, s])] if not np.isnan(X[:, 0, s]).all() else [np.nan, np.nan]
            for s in range(X.shape[2])
        ])
        print(f"Baricentros calculados: {barycenters}")
        return barycenters

    def align_slices(X, Y, barycenters):
        """Alinha as fatias com base na regressão linear dos baricentros."""
        print("Iniciando alinhamento de fatias...")
        num_slices = X.shape[2]
        #Posição de cada fatia ao longo do eixo Z
        z = np.arange(num_slices).reshape(-1, 1)

        valid_idx = ~np.isnan(barycenters[:, 0])
        print(f"Índices válidos para regressão: {valid_idx}")
        if valid_idx.sum() > 1:
            print("Executando regressão linear...")
            modelX = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 0])
            modelY = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 1])

            estX = modelX.predict(z)
            estY = modelY.predict(z)
            print(f"Estimativas de X: {estX}")
            print(f"Estimativas de Y: {estY}")

            for s in range(num_slices):
                if not np.isnan(X[:, 0, s]).all():
                    shiftX = barycenters[s, 0] - estX[s]
                    shiftY = barycenters[s, 1] - estY[s]
                    print(f"Aplicando deslocamento na fatia {s}: shiftX={shiftX}, shiftY={shiftY}")
                    X[:, :, s] -= shiftX
                    Y[:, :, s] -= shiftY

        return X, Y

    # Calcular os baricentros
    endo_barycenters = calculate_barycenters(endoX, endoY)
    epi_barycenters = calculate_barycenters(epiX, epiY)

    # Alinhar as estruturas
    endoX, endoY = align_slices(endoX, endoY, endo_barycenters)
    epiX, epiY = align_slices(epiX, epiY, epi_barycenters)

    print("Alinhamento completo.")

    # Atualizar a estrutura original com as coordenadas alinhadas
    setstruct.EndoX = endoX
    setstruct.EndoY = endoY
    setstruct.EpiX = epiX
    setstruct.EpiY = epiY

    # Salvar o arquivo processado
    output_filename = 'analise.mat'
    savemat(output_filename, {'setstruct': setstruct}, do_compression=True)
    print(f"Arquivo alinhado salvo como: {output_filename}")

    return data

# Exemplo de uso
read_mat('Patient_1.mat')
