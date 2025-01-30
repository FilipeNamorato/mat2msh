from scipy.io import loadmat
import numpy as np
import os

# Para usar DBSCAN, instale scikit-learn se ainda não tiver: pip install scikit-learn
from sklearn.cluster import DBSCAN

def readScar():
    """
    Lê os ROIs do arquivo .mat e retorna um array de pontos (X, Y, Z).
    Mantém a lógica de 'fatias', mas ao final também gera um array unificado.
    """
    mat_filename = "Patient_1_new.mat"
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)  # Carrega o arquivo
    setstruct = data['setstruct']
    
    # Acessando o campo 'Roi' e os subcampos dinamicamente
    rois = setstruct[0][0]['Roi']

    # fatias[z] -> lista de (x, y) para aquele Z
    fatias = {}

    for idx, roi in enumerate(rois):  # Iterar sobre os ROIs do arquivo
        print(f"Processando ROI {idx+1}...")
        try:
            # Extração das coordenadas X, Y e Z, se existirem
            x_coords = roi['X'] if 'X' in roi.dtype.names else None
            y_coords = roi['Y'] if 'Y' in roi.dtype.names else None
            z_values = roi['Z'] if 'Z' in roi.dtype.names else None

            if x_coords is None or y_coords is None or z_values is None:
                print(f"Erro: Dados incompletos para ROI {idx+1}")
                continue

            # Cada ROI pode conter várias fatias (slices), então iteramos sobre elas
            for i in range(len(z_values)):
                x_arr = x_coords[i].flatten()
                y_arr = y_coords[i].flatten()
                z_val = float(z_values[i][0][0])  # Extrai o valor escalar de Z (uma fatia)

                if len(x_arr) == 0 or len(y_arr) == 0:
                    print(f"Erro: Fatia {z_val} do ROI {idx+1} está vazia.")
                    continue

                # Se ainda não houver uma lista para esse valor de Z, criamos uma
                if z_val not in fatias:
                    fatias[z_val] = []
                
                # Adicionar as coordenadas X, Y na lista do dicionário correspondente a esse Z
                fatias[z_val].extend(zip(x_arr, y_arr))
                print(f"Adicionado ROI {idx+1}, Fatia {z_val}, Pontos={len(x_arr)}")

        except Exception as e:
            print(f"Erro ao processar ROI {idx+1}: {e}")

    # Agora criar um array unificado de (X, Y, Z) para uso na clusterização em 3D
    pontos_3d = []
    for z, coords in fatias.items():
        for (x, y) in coords:
            pontos_3d.append([x, y, z])

    # Converte a lista para um array NumPy para melhor desempenho e uso na clusterização
    pontos_3d = np.array(pontos_3d)  # shape (N, 3)

    return fatias, pontos_3d

def save_fatias_to_txt(fatias, output_dir="fatias"):
    """
    Salva coordenadas X, Y em arquivos TXT separados por fatia (Z).

    Cada arquivo recebe o nome 'fatia_<Z>.txt' e conterá as colunas:
    X,Y
    10.0,20.0
    15.0,25.0
    ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for z, coordenadas in fatias.items():
        filename = os.path.join(output_dir, f"fatia_{int(z)}.txt")
        print(f"Salvando fatia {z} no arquivo: {filename}")

        with open(filename, "w") as file:
            # Primeiro escrevemos o cabeçalho
            file.write("X,Y\n")
            for x, y in coordenadas:
                file.write(f"{x},{y}\n")

    print("Arquivos de fatias gerados com sucesso!")

def cluster_scar(pontos_3d, eps=2.0, min_samples=5):
    """
    Aplica DBSCAN nos pontos 3D para agrupar 'scars' próximas.
    
    Parâmetros:
    - eps: raio máximo de distância entre dois pontos para que sejam considerados vizinhos
    - min_samples: número mínimo de pontos para que um conjunto seja definido como um cluster
    
    Retorna:
    - dicionário: {cluster_id: lista_de_pontos (X,Y,Z)}

    Observação:
    - Os pontos marcados como -1 (outliers) são ignorados, pois não se encaixam em nenhum cluster.
    - Ajustar eps e min_samples dependendo da escala dos dados e da densidade desejada.
    """
    if len(pontos_3d) == 0:
        print("Nenhum ponto para clusterizar.")
        return {}

    # Instancia o algoritmo DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    # Ajusta e prediz o rótulo de cada ponto (labels) com base na densidade
    labels = db.fit_predict(pontos_3d)

    clusters = {}
    for i, lbl in enumerate(labels):
        # lbl = -1 indica que o ponto foi marcado como outlier
        if lbl == -1:
            continue
        # Cria uma nova lista para o cluster se ainda não existir
        if lbl not in clusters:
            clusters[lbl] = []
        # Adiciona o ponto ao cluster correspondente
        clusters[lbl].append(pontos_3d[i])

    print(f"Encontrados {len(clusters)} clusters (excluindo outliers).")
    return clusters

def save_clusters_to_txt(clusters, output_dir="clusters_dbscan"):
    """
    Salva cada cluster em um arquivo separado dentro de 'output_dir'.
    O formato de cada arquivo será:
    X,Y,Z
    10.0,20.0,5.0
    15.0,25.0,5.0
    ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for lbl, pts in clusters.items():
        filename = os.path.join(output_dir, f"cluster_{lbl}.txt")
        with open(filename, "w") as file:
            file.write("X,Y,Z\n")
            for x, y, z in pts:
                file.write(f"{x},{y},{z}\n")
        print(f"Cluster {lbl} salvo em: {filename}")

if __name__ == "__main__":
    # Ler os dados das ROIs (agrupados por fatias e também num array 3D unificado)
    fatias_data, pontos_3d = readScar()

    # Salvar as fatias separadamente em arquivos .txt
    save_fatias_to_txt(fatias_data, "fatias_txt")

    # Aplicar DBSCAN para agrupar as scars em 3D com parâmetros padrão
    clusters = cluster_scar(pontos_3d, eps=2.0, min_samples=5)

    # Salvar cada cluster resultante em um arquivo de texto individual
    save_clusters_to_txt(clusters, "clusters_dbscan")
