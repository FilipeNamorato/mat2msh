import numpy as np
from scipy.io import loadmat, savemat

def read_mat(mat_filename):
    # Carregar o arquivo MATLAB
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']

    # Extrair as coordenadas de cada estrutura e verificar se são válidas
    def get_coordinates(field):
        coords = getattr(setstruct, field, None)
        if coords is None or len(coords.shape) < 2:
            raise ValueError(f"Coordenadas inválidas para {field}.")
        if len(coords.shape) == 2:
            coords = coords[:, :, np.newaxis]  # Adiciona dimensão extra se necessário
        return coords

    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    RVEndoX = get_coordinates('RVEndoX')
    RVEndoY = get_coordinates('RVEndoY')
    RVEpiX = get_coordinates('RVEpiX')
    RVEpiY = get_coordinates('RVEpiY')

    def align_structure(structX, structY, reference_slice):
        """Alinha uma estrutura em relação a uma fatia de referência."""
        num_slices = structX.shape[2]  # Verifica se há a dimensão esperada

        # Verifica se a fatia de referência é válida
        refX_mean = np.nanmean(structX[:, 0, reference_slice])
        refY_mean = np.nanmean(structY[:, 0, reference_slice])

        # Alinha cada fatia em relação à fatia de referência
        for s in range(num_slices):
            if np.isnan(structX[:, 0, s]).all():
                print(f"Fatia {s} contém apenas NaNs, pulando...")
                continue

            shiftX = np.nanmean(structX[:, 0, s]) - refX_mean
            shiftY = np.nanmean(structY[:, 0, s]) - refY_mean

            print(f"Fatia {s}: shiftX acumulado={shiftX}, shiftY acumulado={shiftY}")

            structX[:, 0, s] -= shiftX
            structY[:, 0, s] -= shiftY

    # Escolher a fatia de referência
    reference_slice = 4

    # Alinhar as estruturas
    align_structure(endoX, endoY, reference_slice)
    align_structure(RVEndoX, RVEndoY, reference_slice)
    align_structure(RVEpiX, RVEpiY, reference_slice)

    print("Alinhamento completo para todas as estruturas.")

    # Atualizar as coordenadas dentro da estrutura original
    setstruct.EndoX = endoX
    setstruct.EndoY = endoY
    setstruct.RVEndoX = RVEndoX
    setstruct.RVEndoY = RVEndoY
    setstruct.RVEpiX = RVEpiX
    setstruct.RVEpiY = RVEpiY

    # Salvar o arquivo modificado
    output_filename = 'analise.mat'
    savemat(output_filename, {'setstruct': setstruct}, do_compression=True)
    print(f"Arquivo alinhado salvo como: {output_filename}")

    return data  # Retorna os dados para uso adicional, se necessário
