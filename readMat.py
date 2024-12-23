import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LinearRegression

def read_mat(mat_filename, RVyes=False):
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

    # Obter valores de SliceThickness e SliceGap
    slice_thickness = getattr(setstruct, 'SliceThickness', None)
    slice_gap = getattr(setstruct, 'SliceGap', 0.0)  # Default para 0 se não existir
    print(slice_gap)

    if slice_thickness is None:
        raise ValueError("Campo 'SliceThickness' não encontrado no arquivo .mat")

    # Calcular Z baseado em SliceThickness e SliceGap
    def calculate_z(num_slices, slice_thickness, slice_gap, num_points_per_slice):
        z_values = np.arange(num_slices) * slice_thickness
        return np.tile(z_values, (num_points_per_slice, 1))

    # Extrair as coordenadas das estruturas
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    epiX = get_coordinates('EpiX')
    epiY = get_coordinates('EpiY')

    num_slices = epiX.shape[2]
    num_points_per_slice = epiX.shape[0]

    # Gerar valores de Z para todas as fatias
    EndoZ = calculate_z(num_slices, slice_thickness, slice_gap, endoX.shape[0])
    EpiZ = calculate_z(num_slices, slice_thickness, slice_gap, epiX.shape[0])

    def calculate_barycenters(endoX, endoY, epiX, epiY):
        """Calcula os baricentros para cada fatia, considerando NaNs em EndoX."""
        barycenters = np.zeros((epiX.shape[2], 2))
        for s in range(epiX.shape[2]):
            if not np.isnan(endoX[0, 0, s]):  # Se EndoX não tem NaN para essa fatia
                barycenters[s, 0] = 0.5 * np.nanmean(epiX[:, :, s] + endoX[:, :, s])
                barycenters[s, 1] = 0.5 * np.nanmean(epiY[:, :, s] + endoY[:, :, s])
            else:
                barycenters[s, 0] = np.nanmean(epiX[:, :, s])
                barycenters[s, 1] = np.nanmean(epiY[:, :, s])
        return barycenters

    def align_slices(X, Y, barycenters, z_values):
        """Alinha as fatias com base na regressão linear dos baricentros."""
        num_slices = X.shape[2]
        z = z_values.reshape(-1, 1)

        valid_idx = ~np.isnan(barycenters[:, 0])
        if valid_idx.sum() > 1:
            modelX = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 0])
            modelY = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 1])

            estX = modelX.predict(z)
            estY = modelY.predict(z)

            for s in range(num_slices):
                if not np.isnan(X[:, 0, s]).all():
                    shiftX = barycenters[s, 0] - estX[s]
                    shiftY = barycenters[s, 1] - estY[s]
                    X[:, :, s] -= shiftX
                    Y[:, :, s] -= shiftY

        return X, Y

    # Calcular os baricentros e alinhar Endo e Epi
    endo_barycenters = calculate_barycenters(endoX, endoY, epiX, epiY)
    endoX, endoY = align_slices(endoX, endoY, endo_barycenters, EndoZ[0])
    epiX, epiY = align_slices(epiX, epiY, endo_barycenters, EpiZ[0])

    if RVyes:
        # Alinhamento do RV
        RVEndoX = get_coordinates('RVEndoX')
        RVEndoY = get_coordinates('RVEndoY')
        RVEpiX = get_coordinates('RVEpiX')
        RVEpiY = get_coordinates('RVEpiY')

        RVEndoZ = calculate_z(num_slices, slice_thickness, slice_gap, RVEndoX.shape[0])
        RVEpiZ = calculate_z(num_slices, slice_thickness, slice_gap, RVEpiX.shape[0])

        rv_barycenters = calculate_barycenters(RVEndoX, RVEndoY, RVEpiX, RVEpiY)

        # Alinhar as fatias de RV Endo e RV Epi
        RVEndoX, RVEndoY = align_slices(RVEndoX, RVEndoY, rv_barycenters, RVEndoZ[0])
        RVEpiX, RVEpiY = align_slices(RVEpiX, RVEpiY, rv_barycenters, RVEpiZ[0])

        # Atualizar setstruct com as coordenadas alinhadas do RV
        setstruct.RVEndoX = RVEndoX
        setstruct.RVEndoY = RVEndoY
        setstruct.RVEpiX = RVEpiX
        setstruct.RVEpiY = RVEpiY

    print("Alinhamento completo.")

    # Atualizar a estrutura original com as coordenadas alinhadas
    setstruct.EndoX = endoX
    setstruct.EndoY = endoY
    setstruct.EpiX = epiX
    setstruct.EpiY = epiY

    # Salvar o arquivo processado
    output_filename = 'analise_alinhada.mat'
    savemat(output_filename, {'setstruct': setstruct}, do_compression=True)
    print(f"Arquivo alinhado salvo como: {output_filename}")

    return data

# Exemplo de uso
read_mat('Patient_1.mat', RVyes=True)
