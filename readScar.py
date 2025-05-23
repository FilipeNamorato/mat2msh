import sys, os, glob, subprocess, argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import vtk
import meshio
import time

# Estrutura simples para cada ROI em uma fatia: nome, índice Z e lista de pontos (x,y)
ROIEntry = namedtuple('ROIEntry', ['name', 'z', 'points'])

################################
# 1) Leitura do .mat e extração de ROIs
################################
def readScar(mat_filename):
    """
    Lê o arquivo .mat e retorna uma lista plana de ROIEntry.
    Cada ROIEntry contém:
      - name: nome da ROI (ex: "ROI-1")
      - z: índice da fatia onde essa ROI foi anotada
      - points: lista de tuplas (x,y) dos pontos da ROI

    Explicação:
    - "setstruct" é a estrutura MATLAB com metadados e ROIs.
    - Extraímos de cada ROI seus arrays X, Y, Z e o nome para cada elemento.
    - Para cada sub-fat ia (índice i), associamos o nome correto e empacotamos
      os pontos convertidos em Python.
    """
    print(f"Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename)
    rois = data['setstruct'][0][0]['Roi']

    entries = []
    for idx, roi in enumerate(rois):
        # Extrai lista de nomes para cada sub-fat ia
        raw_names = roi['Name'].flatten()
        slice_names = []
        for element in raw_names:
            # Converte array numpy para string
            if isinstance(element, np.ndarray):
                val = element.flat[0]
            else:
                val = element
            slice_names.append(str(val).strip())
        # Arrays de coordenadas
        X = roi['X']; Y = roi['Y']; Z = roi['Z']
        # Para cada sub-fatia, empacota um ROIEntry
        for i in range(len(Z)):
            z_val = int(np.atleast_1d(Z[i]).flat[0])
            x_arr = np.atleast_1d(X[i]).flatten()
            y_arr = np.atleast_1d(Y[i]).flatten()
            if x_arr.size == 0 or y_arr.size == 0:
                continue  # sem pontos nesta sub-fatia
            name = slice_names[i] if i < len(slice_names) else slice_names[0]
            pts = list(zip(x_arr, y_arr))
            entries.append(ROIEntry(name, z_val, pts))
    return entries

################################
# 2) Agrupamento de ROIs por fatia
################################
def group_by_slice(entries):
    """
    Recebe lista de ROIEntry e retorna:
      { z: { roi_name: [ (x,y), ... ], ... }, ... }
    """
    fatias = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias[e.z][e.name].extend(e.points)
    return fatias

################################
# 3) Visualização 2D das fatias
################################
def plot_slices(fatias):
    """
    Para conferência, plota cada fatia mostrando:
      - Pontos de cada ROI coloridos
      - Nome e centróide marcado
    """
    for z, roi_map in sorted(fatias.items()):
        plt.figure(figsize=(6,6))
        for name, pts in roi_map.items():
            arr = np.array(pts)
            plt.scatter(arr[:,0], arr[:,1], label=name, s=20)
            cen = arr.mean(axis=0)
            plt.text(cen[0], cen[1], name,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', alpha=0.5))
        plt.title(f"Slice Z={z}")
        plt.gca().invert_yaxis()
        plt.legend(fontsize=7)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

################################
# 4) Alinhamento e gravação de fatias em .txt
################################
def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir="fatias"):
    """
    Aplica deslocamentos (shifts) em X/Y para cada fatia e salva em "fatias/".
    Arquivos: fatia_<z>.txt contendo linhas "x_alinhado y_alinhado z".
    """
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)
    for z, roi_map in sorted(fatias.items()):
        # Seleciona shift adequado ou zero se fora do alcance
        sx = shifts_x[z] if 0 <= z < len(shifts_x) else 0
        sy = shifts_y[z] if 0 <= z < len(shifts_y) else 0
        fname = os.path.join(output_dir, f"fatia_{z}.txt")
        with open(fname, 'w') as f:
            for pts in roi_map.values():
                for x, y in pts:
                    f.write(f"{x - sx} {y - sy} {z}\n")
        print(f"Saved slice {z} to {fname}")

################################
 # 5) Salva ROIs separados em .txt
################################

def save_rois_extruded_to_txt(fatias, mat_filename, output_dir="rois_extruded", num_layers=1):

    data = loadmat(mat_filename)
    ss = data['setstruct']
    slice_thickness = float(ss['SliceThickness'][0][0][0][0])
    gap             = float(ss['SliceGap'][0][0][0][0])
    resolution_x    = float(ss['ResolutionX'][0][0][0][0])
    resolution_y    = float(ss['ResolutionY'][0][0][0][0])
    dz = slice_thickness + gap

    os.makedirs(output_dir, exist_ok=True)

    for z, roi_map in sorted(fatias.items()):
        for roi_name, points in roi_map.items():
            safe_name = roi_name.replace(" ", "_").replace("/", "_")
            fname = os.path.join(output_dir, f"roi_{safe_name}_z{z}.txt")
            with open(fname, 'w') as f:
                z_base = z * dz
                z_top  = z_base + dz

                # Gera N camadas entre z_base e z_top
                for layer in range(num_layers + 1):
                    alpha = layer / num_layers #QTD de subfatias
                    # Interpola entre z_base e z_top
                    z_interp = z_base * (1 - alpha) + z_top * alpha
                    for x, y in points:
                        x_out = x * resolution_x
                        y_out = y * resolution_y
                        f.write(f"{x_out:.6f} {y_out:.6f} {z_interp:.6f}\n")
            print(f"Saved extruded ROI '{roi_name}' (slice {z}) to: {fname}")

################################
# 6) Geração de superfícies (.ply) e STL
################################
def generate_surfaces_and_stl():
    """
    Para cada arquivo roi_<nome>_z<z>.txt em rois_extruded/:
      1) chama make_surface.py para gerar .ply
      2) converte .ply em .stl usando PlyToStl
    """
    # prepara diretórios de saída
    os.makedirs("./output/plyFiles",  exist_ok=True)
    os.makedirs("./output/scarFiles", exist_ok=True)

    # encontra todos os ROIs extrudados
    txts = sorted(glob.glob("./rois_extruded/roi_*.txt"))
    for txt in txts:
        base = os.path.splitext(os.path.basename(txt))[0]  # ex: "roi_ROI-1_z11"
        ply = f"./output/plyFiles/{base}.ply"
        stl = f"./output/scarFiles/{base}.stl"

        # 1) gera o PLY
        try:
            subprocess.run(
                f"python3 make_surface.py {txt} --cover-both-ends",
                shell=True, check=True
            )
            print(f"Surface for {txt} generated at {ply}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating PLY for {txt}: {e}")
            continue

        # 2) converte PLY → STL
        if os.path.exists(ply):
            try:
                subprocess.run(
                    f"./convertPly2STL/build/PlyToStl {ply} {stl} 1",
                    shell=True, check=True
                )
                print(f"STL created: {stl}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {ply} to STL: {e}")
        else:
            print(f"PLY file not found for {txt}, skipping STL conversion.")


################################
# MAIN: execução completa
################################
def main():
    parser = argparse.ArgumentParser(description="Full scar pipeline")
    parser.add_argument('matfile', help='Path to .mat file')
    parser.add_argument('--shiftx', default='endo_shifts_x.txt')
    parser.add_argument('--shifty', default='endo_shifts_y.txt')
    args = parser.parse_args()

    # 1-3: leitura, agrupamento e plot
    entries = readScar(args.matfile)
    fatias = group_by_slice(entries)
    plot_slices(fatias)
    # 4: grava fatias alinhadas
    save_fatias_to_txt(fatias, args.shiftx, args.shifty)

    # 5) Nova etapa: extrusão simples de cada ROI
    save_rois_extruded_to_txt(fatias, args.matfile, output_dir="rois_extruded")

    # 6) Geração de superfícies e STL a partir das extrusões
    generate_surfaces_and_stl()

if __name__ == '__main__':
    main()
