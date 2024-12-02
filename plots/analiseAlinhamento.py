import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read_coordinates(mat_filename):
    """
    Lê as coordenadas das estruturas de um arquivo .mat.
    """
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']
    
    def get_coordinates(field):
        coords = getattr(setstruct, field, None)
        if coords is None:
            raise ValueError(f"Campo {field} não encontrado.")
        if len(coords.shape) == 2:
            coords = coords[:, np.newaxis, :]  # Insere uma dimensão intermediária (1)
        return coords

    # Extrair as coordenadas das estruturas
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    
    return endoX, endoY

def analyze_alignment(original_filename, processed_filename):
    """
    Compara o alinhamento entre o arquivo original e o pós-processado.
    """
    # Carrega coordenadas do arquivo original e do pós-processado
    endoX_original, endoY_original = read_coordinates(original_filename)
    endoX_aligned, endoY_aligned = read_coordinates(processed_filename)

    num_slices = endoX_original.shape[2]
    
    # Calcula o deslocamento ponto a ponto entre fatias adjacentes
    def calculate_displacement(structX, structY):
        displacements = []
        for s in range(1, num_slices):
            valid_points = ~np.isnan(structX[:, 0, s]) & ~np.isnan(structX[:, 0, s - 1])
            if np.sum(valid_points) > 0:
                displacement = np.mean(np.sqrt(
                    (structX[valid_points, 0, s] - structX[valid_points, 0, s - 1]) ** 2 +
                    (structY[valid_points, 0, s] - structY[valid_points, 0, s - 1]) ** 2
                ))
                displacements.append(displacement)
        return displacements
    
    # Calcula deslocamentos antes e depois do alinhamento
    original_displacement = calculate_displacement(endoX_original, endoY_original)
    aligned_displacement = calculate_displacement(endoX_aligned, endoY_aligned)

    # Cálculo das variâncias
    var_original = np.var(original_displacement)
    var_aligned = np.var(aligned_displacement)
    
    # Plot dos histogramas para análise visual
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(original_displacement, bins=20, alpha=0.7, label='Original')
    plt.hist(aligned_displacement, bins=20, alpha=0.7, label='Alinhado')
    plt.xlabel('Deslocamento médio entre fatias')
    plt.ylabel('Frequência')
    plt.legend()
    plt.title('Distribuição dos deslocamentos entre fatias')
    
    plt.subplot(1, 2, 2)
    plt.plot(original_displacement, label="Original")
    plt.plot(aligned_displacement, label="Alinhado")
    plt.xlabel("Fatia")
    plt.ylabel("Deslocamento médio")
    plt.legend()
    plt.title("Deslocamento médio entre fatias antes e depois do alinhamento")
    plt.tight_layout()
    plt.show()
    
    # Avaliação final
    print(f"Variância do deslocamento antes do alinhamento: {var_original:.3f}")
    print(f"Variância do deslocamento após o alinhamento: {var_aligned:.3f}")
    
    if var_aligned < var_original:
        print("O alinhamento parece ter sido bem-sucedido, com redução na variância dos deslocamentos entre fatias.")
    else:
        print("O alinhamento pode não ter sido eficaz; considere revisar o processo.")

# Exemplo de uso com os arquivos .mat original e pós-processado
original_filename = 'Patient_1.mat'
processed_filename = 'analise.mat'
analyze_alignment(original_filename, processed_filename)
