import numpy as np
from scipy.io import loadmat, savemat

def read_mat(mat_filename):
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']

    def get_coordinates(field):
        coords = getattr(setstruct, field, None)
        if coords is None:
            raise ValueError(f"Campo {field} não encontrado.")
        if len(coords.shape) == 2:
            coords = coords[:, np.newaxis, :]
        return coords

    # Extrair as coordenadas das estruturas
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    RVEndoX = get_coordinates('RVEndoX')
    RVEndoY = get_coordinates('RVEndoY')
    RVEpiX = get_coordinates('RVEpiX')
    RVEpiY = get_coordinates('RVEpiY')

    def align_with_neighbors(structX, structY, min_points=5):
        """
        Alinha cada fatia com base nas fatias anterior e posterior.
        Apenas pontos válidos (não NaN) são considerados.
        """
        num_slices = structX.shape[2]
        alignedX = np.copy(structX)
        alignedY = np.copy(structY)

        for s in range(1, num_slices - 1):
            # Verificar pontos válidos nas fatias vizinhas
            valid_prev = ~np.isnan(structX[:, 0, s - 1]) & ~np.isnan(structX[:, 0, s])
            valid_next = ~np.isnan(structX[:, 0, s + 1]) & ~np.isnan(structX[:, 0, s])

            if np.sum(valid_prev) >= min_points:
                shiftX_prev = np.nanmean(structX[valid_prev, 0, s] - structX[valid_prev, 0, s - 1])
                shiftY_prev = np.nanmean(structY[valid_prev, 0, s] - structY[valid_prev, 0, s - 1])
            else:
                shiftX_prev, shiftY_prev = 0, 0  # Sem deslocamento

            if np.sum(valid_next) >= min_points:
                shiftX_next = np.nanmean(structX[valid_next, 0, s + 1] - structX[valid_next, 0, s])
                shiftY_next = np.nanmean(structY[valid_next, 0, s + 1] - structY[valid_next, 0, s])
            else:
                shiftX_next, shiftY_next = 0, 0  # Sem deslocamento

            # Média dos deslocamentos anterior e posterior
            shiftX = (shiftX_prev + shiftX_next) / 2
            shiftY = (shiftY_prev + shiftY_next) / 2

            # Aplicar deslocamento apenas nos pontos válidos
            valid_current = ~np.isnan(structX[:, 0, s])
            alignedX[valid_current, 0, s] -= shiftX
            alignedY[valid_current, 0, s] -= shiftY

        return alignedX, alignedY

    # Aplicar o alinhamento às estruturas
    endoX, endoY = align_with_neighbors(endoX, endoY)
    RVEndoX, RVEndoY = align_with_neighbors(RVEndoX, RVEndoY)
    RVEpiX, RVEpiY = align_with_neighbors(RVEpiX, RVEpiY)

    print("Alinhamento com vizinhos completo.")

    # Atualizar a estrutura original com as coordenadas alinhadas
    setstruct.EndoX = endoX
    setstruct.EndoY = endoY
    setstruct.RVEndoX = RVEndoX
    setstruct.RVEndoY = RVEndoY
    setstruct.RVEpiX = RVEpiX
    setstruct.RVEpiY = RVEpiY

    # Salvar o arquivo processado
    output_filename = 'analise.mat'
    savemat(output_filename, {'setstruct': setstruct}, do_compression=True)
    print(f"Arquivo alinhado salvo como: {output_filename}")

    return data
