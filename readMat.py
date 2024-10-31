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
            coords = coords[:, np.newaxis, :]  # Inserir dimensão intermediária (1)
        return coords

    # Extrair as coordenadas das estruturas
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    RVEndoX = get_coordinates('RVEndoX')
    RVEndoY = get_coordinates('RVEndoY')
    RVEpiX = get_coordinates('RVEpiX')
    RVEpiY = get_coordinates('RVEpiY')

    def align_slices(structX, structY, min_points=5, window_size=20):
        num_slices = structX.shape[2]
        alignedX = np.copy(structX)
        alignedY = np.copy(structY)
        
        def calculate_shift(prev, current, next_slice):
            valid_prev = ~np.isnan(prev) & ~np.isnan(current)
            valid_next = ~np.isnan(current) & ~np.isnan(next_slice)
            
            shift_prev = np.nanmean(current[valid_prev] - prev[valid_prev]) if np.sum(valid_prev) >= min_points else 0
            shift_next = np.nanmean(next_slice[valid_next] - current[valid_next]) if np.sum(valid_next) >= min_points else 0
            
            return (shift_prev + shift_next) / 2

        for s in range(1, num_slices - 1):
            window_start = max(0, s - window_size // 2)
            window_end = min(num_slices, s + window_size // 2 + 1)
            
            shiftsX = []
            shiftsY = []
            for w in range(window_start, window_end - 1):
                shiftX = calculate_shift(structX[:, 0, w], structX[:, 0, w + 1], structX[:, 0, w + 2] if w + 2 < num_slices else structX[:, 0, w + 1])
                shiftY = calculate_shift(structY[:, 0, w], structY[:, 0, w + 1], structY[:, 0, w + 2] if w + 2 < num_slices else structY[:, 0, w + 1])
                shiftsX.append(shiftX)
                shiftsY.append(shiftY)
            
            mean_shiftX = np.mean(shiftsX)
            mean_shiftY = np.mean(shiftsY)
            
            valid_current = ~np.isnan(structX[:, 0, s])
            alignedX[valid_current, 0, s] -= mean_shiftX
            alignedY[valid_current, 0, s] -= mean_shiftY

        return alignedX, alignedY

    # Aplicar o alinhamento às estruturas usando a nova função
    endoX, endoY = align_slices(endoX, endoY)
    RVEndoX, RVEndoY = align_slices(RVEndoX, RVEndoY)
    RVEpiX, RVEpiY = align_slices(RVEpiX, RVEpiY)

    print("Alinhamento com vizinhos completo.")

    # Atualizar a estrutura original com as coordenadas corrigidas
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

# Exemplo de uso
read_mat('Patient_1.mat')
