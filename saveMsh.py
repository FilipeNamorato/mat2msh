import numpy as np
import os
from datetime import datetime
from scipy.io import loadmat

def save_structures_to_txt(mat_filename, output_dir):
    """
    Salva as coordenadas de estruturas (LVEndo, LVEpi, RVEndo, RVEpi) em arquivos .txt únicos,
    contendo todas as fatias de cada estrutura no formato MATLAB.

    Parameters:
    - mat_filename: Caminho para o arquivo .mat.
    - output_dir: Diretório onde os arquivos serão salvos.
    """
    # Cria um diretório de saída baseado na data atual
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, date_str)
    os.makedirs(output_path, exist_ok=True)

    try:
        data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
        setstruct = data['setstruct']
        slice_thickness = getattr(setstruct, 'SliceThickness', 8.0)  # Obtém SliceThickness ou usa valor padrão
        slice_gap = getattr(setstruct, 'SliceGap', 0.0)  # Obtém SliceGap ou usa valor padrão
    except Exception as e:
        print(f"Erro ao carregar o arquivo .mat: {e}")
        return None

    # Estruturas a serem processadas
    structures = {
        'LVEndo': ('EndoX', 'EndoY'),
        'LVEpi': ('EpiX', 'EpiY'),
        'RVEndo': ('RVEndoX', 'RVEndoY'),
        'RVEpi': ('RVEpiX', 'RVEpiY')
    }

    # Processar cada estrutura
    for name, (x_attr, y_attr) in structures.items():
        try:
            x_coords = getattr(setstruct, x_attr)
            y_coords = getattr(setstruct, y_attr)

            num_slices = x_coords.shape[2] if x_coords.ndim == 3 else x_coords.shape[1]

            # Nome do arquivo de saída
            output_filename = os.path.join(output_path, f"Patient_1-{name}.txt")
            with open(output_filename, 'w') as f:
                for s in range(num_slices):
                    # Obter as coordenadas da fatia
                    x_slice = x_coords[:, 0, s] if x_coords.ndim == 3 else x_coords[:, s]
                    y_slice = y_coords[:, 0, s] if y_coords.ndim == 3 else y_coords[:, s]

                    # Filtrar pontos válidos
                    valid_mask = ~np.isnan(x_slice) & ~np.isnan(y_slice)
                    valid_x = x_slice[valid_mask]
                    valid_y = y_slice[valid_mask]
                    z_value = np.full(valid_x.shape, s * (slice_thickness + slice_gap))  # Ajusta Z com SliceThickness e SliceGap

                    # Escrever coordenadas válidas no arquivo
                    coords = np.column_stack((valid_x, valid_y, z_value))
                    np.savetxt(f, coords, fmt="%.6f", delimiter=" ")

                print(f"Arquivo {output_filename} salvo com sucesso.")

        except AttributeError:
            print(f"Erro: {x_attr} ou {y_attr} não encontrado no arquivo {mat_filename}")

    print("Exportação concluída com sucesso.")
    return output_path

def main():
    mat_filename = "./analise_alinhada.mat"
    output_dir = "saida"

    if not os.path.exists(mat_filename):
        print(f"Erro: O arquivo {mat_filename} não existe.")
        return

    output_txt = save_structures_to_txt(mat_filename, output_dir)
    if not output_txt:
        print("Erro durante a exportação para .txt.")
        return

    print(f"Arquivos exportados para o diretório: {output_txt}")

if __name__ == "__main__":
    main()
