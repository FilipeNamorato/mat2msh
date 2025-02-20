from scipy.io import loadmat
import numpy as np
import os
import glob
import subprocess
# pip install scikit-learn
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

    fatias = {}  # fatias[z] -> lista de (x, y) para aquele Z

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


                print(f"Adicionado ROI {idx+1}, Fatia {z_val}, "
                      f"Pontos={len(x_arr)}")

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
    Salva coordenadas X, Y das fibroses em arquivos TXT separados por fatia (Z),
    aplicando o deslocamento do alinhamento do miocárdio.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carregar os deslocamentos aplicados ao coração
    try:
        endo_shifts_x = np.loadtxt("endo_shifts_x.txt")
        endo_shifts_y = np.loadtxt("endo_shifts_y.txt")
    except Exception as e:
        print(f"Erro ao carregar deslocamentos: {e}")
        return

    for z, coordenadas in fatias.items():
        filename = os.path.join(output_dir, f"fatia_{int(z)}.txt")
        print(f"Salvando fatia {z} no arquivo: {filename}")

        with open(filename, "w") as file:
            slice_idx = int(z)  # Assumindo que o valor de Z representa o índice da fatia
            
            # Verificar se o índice da fatia está dentro do intervalo dos deslocamentos
            if 0 <= slice_idx < len(endo_shifts_x):
                shift_x = endo_shifts_x[slice_idx]
                shift_y = endo_shifts_y[slice_idx]
            else:
                shift_x = 0
                shift_y = 0
                print(f"Atenção: Fatia {slice_idx} fora do intervalo dos deslocamentos!")

            # Escreve todos os pontos da fibrose com o deslocamento aplicado
            for x, y in coordenadas:
                x_aligned = x - shift_x
                y_aligned = y - shift_y
                file.write(f"{x_aligned} {y_aligned} {z}\n")

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
    Saves each cluster in a separate .txt file inside 'output_dir'.

    For each cluster, it rescales:
      x by resolution_x
      y by resolution_y
      z by slice_thickness

    Format of each output file (no header):
      x y z
      10.0 20.0 5.0
      15.0 25.0 5.0
      ...
    """

    mat_filename = "Patient_1_new.mat"
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename)  # Loads the .mat file
    setstruct = data['setstruct']

    # Extract slice thickness (assuming it's stored like this: setstruct['SliceThickness'][0][0][0][0])
    slice_thickness = setstruct['SliceThickness'][0][0][0][0]
    slice_gap = setstruct['SliceGap'][0][0][0][0]  # Default to 0 if not present
    resolution_x = setstruct['ResolutionX'][0][0][0][0]
    resolution_y = setstruct['ResolutionY'][0][0][0][0]
    print("-------------------------------------------------------------------------------")
    print(resolution_x, resolution_y)
    print("-------------------------------------------------------------------------------")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each cluster in the dictionary
    for lbl, pts in clusters.items():
        filename = os.path.join(output_dir, f"cluster_{lbl}.txt")
        with open(filename, "w") as file:
            for x, y, z in pts:
                # Scale each coordinate
                x_scaled = x * resolution_x
                y_scaled = y * resolution_y
                z_scaled = z * (slice_thickness + slice_gap)
                # Write them to the file
                file.write(f"{x_scaled} {y_scaled} {z_scaled}\n")

        print(f"Cluster {lbl} saved to: {filename}")



if __name__ == "__main__":
    # Ler os dados das ROIs (agrupados por fatias e também num array 3D unificado)
    fatias_data, pontos_3d = readScar()

    # Salvar as fatias separadamente em arquivos .txt
    save_fatias_to_txt(fatias_data, "fatias_txt")

    # Aplicar DBSCAN para agrupar as scars em 3D com parâmetros padrão
    clusters = cluster_scar(pontos_3d, eps=2.0, min_samples=5)

    # Salvar cada cluster resultante em um arquivo de texto individual
    # (ex.: cluster_0.txt, cluster_1.txt, cluster_2.txt, etc.)
    save_clusters_to_txt(clusters, "clusters_dbscan")

    # Passo 3: Gera superfícies para TODOS os arquivos cluster_*.txt na pasta clusters_dbscan
    cluster_txt_files = sorted(glob.glob("./clusters_dbscan/cluster_*.txt"))

    for surface_file in cluster_txt_files:
        try:
            # --cover-both-ends é só um exemplo de parâmetro adicional, ajuste conforme necessário
            surface_command = f"python3 make_surface.py {surface_file} --cover-both-ends"
            subprocess.run(surface_command, shell=True, check=True)
            print(f"Superfície gerada para {surface_file}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao gerar superfície para {surface_file}: {e}")

    # Passo 4: Converte cada arquivo .ply (cluster_X.ply) em .stl (cluster_X.stl)
    # A make_surface.py deve gerar os arquivos .ply em ./output/plyFiles/cluster_X.ply
    for surface_file in cluster_txt_files:
        # Extrai o nome do cluster (ex.: "cluster_2")
        cluster_name = os.path.splitext(os.path.basename(surface_file))[0]  # cluster_2

        # Monta o caminho para o .ply que o make_surface.py deve ter criado
        ply_file = f"./output/plyFiles/{cluster_name}.ply"
        # Define o STL de saída
        stl_output = f"./output/{cluster_name}.stl"

        if not os.path.exists(ply_file):
            print(f"Erro: não existe arquivo PLY gerado para {ply_file}. Verifique o make_surface.py.")
            continue

        try:
            ply_to_stl_command = f"./convertPly2STL/build/PlyToStl {ply_file} {stl_output} True"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            print(f"STL gerado com sucesso: {stl_output}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao converter {ply_file} para {stl_output}: {e}")

