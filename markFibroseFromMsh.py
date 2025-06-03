import os
import time
import argparse
import meshio
import numpy as np
import vtk
import matplotlib.pyplot as plt

def mark_fibrosis(path_msh, stl_dir, output_path="saida_com_fibrose.msh", plot_centroids=False):
    # 1) Lê o .msh original
    mesh      = meshio.read(path_msh)
    points    = mesh.points               # (N_pts, 3)
    cells     = mesh.cells
    cell_data = mesh.cell_data

    # 2) Localiza os tetraedros e as tags originais
    tetra_index = next(i for i, c in enumerate(cells) if c.type == "tetra")
    tetra       = cells[tetra_index].data    # array shape = (n_tetra, 4)
    tags        = cell_data["gmsh:physical"][tetra_index]  # shape = (n_tetra,)

    n_tetra = tetra.shape[0]

    # --- 3) Monta lista de nós únicos que aparecem nos tetras ---
    # Usamos flatten + np.unique em vez de repetir cada nó:
    flat_nodes      = tetra.flatten()                     # shape = (n_tetra*4,)
    unique_nodes, inverse_idx = np.unique(flat_nodes, return_inverse=True)
    #  - unique_nodes: índices únicos de 'points' que realmente participam de algum tetra
    #  - inverse_idx: array (n_tetra*4,) de inteiros dizendo, para cada posição em flat_nodes,
    #    qual a posição correspondente dentro de unique_nodes.

    # Agora coords dos nós únicos:
    coords_unique = points[unique_nodes]  # (N_unique, 3)

    # 4) Cria vtkPolyData só com esses nós únicos
    vtk_pts_nodes = vtk.vtkPoints()
    for p in coords_unique:
        vtk_pts_nodes.InsertNextPoint(p)
    input_poly_nodes = vtk.vtkPolyData()
    input_poly_nodes.SetPoints(vtk_pts_nodes)

    # --- 5) Se quiser plotar apenas os centróides, criação abaixo (opcional) ---
    # (Vamos calcular os centróides aqui, mas não rodar select sobre eles ainda)
    centroids = np.mean(points[tetra], axis=1)  # (n_tetra, 3)
    # Para plotar depois, caso plot_centroids=True.

    # 6) Prepara array de tags que será atualizado
    tags_new = tags.copy()  # vamos marcar como “2” (fibrose) os tetras identificados

    # 7) Para cada STL de fibrose, vamos
    #    a) rodar vtkSelectEnclosedPoints sobre o conjunto de N_unique nós
    #    b) rodar vtkSelectEnclosedPoints sobre o conjunto de N_tetra centróides
    #    c) combinar: tetra marcado se (centro dentro) OU (algum vértice dentro)

    # Para não recriar o polydata de centróides toda vez, criamos aqui:
    vtk_pts_cent = vtk.vtkPoints()
    for c in centroids:
        vtk_pts_cent.InsertNextPoint(c)
    input_poly_cent = vtk.vtkPolyData()
    input_poly_cent.SetPoints(vtk_pts_cent)

    for fname in sorted(os.listdir(stl_dir)):
        if not fname.lower().endswith(".stl"):
            continue

        full_path = os.path.join(stl_dir, fname)

        # 8) “Espera adaptativa” para garantir que o STL esteja pronto
        for _ in range(20):  # até ~2 segundos
            try:
                with open(full_path, "r", errors="ignore") as f:
                    lines = [next(f) for _ in range(5)]
                if any("vertex" in line for line in lines):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            print(f"[ERROR] STL {fname} not ready or malformed. Skipping.")
            continue

        print(f"Processing {fname} ...")
        reader = vtk.vtkSTLReader()
        reader.SetFileName(full_path)
        reader.Update()
        fib_surface = reader.GetOutput()

        # 9) Cria um selector para N_unique nós
        selector_nodes = vtk.vtkSelectEnclosedPoints()
        selector_nodes.SetInputData(input_poly_nodes)
        selector_nodes.SetSurfaceData(fib_surface)
        selector_nodes.SetTolerance(1e-6)  # pode ajustar (1e-5, 1e-4) para "folga"
        selector_nodes.Update()

        # 10) Recupera booleano de quais nós estão dentro
        Nuniq = coords_unique.shape[0]
        inside_unique_nodes = np.zeros(Nuniq, dtype=bool)
        for i in range(Nuniq):
            inside_unique_nodes[i] = bool(selector_nodes.IsInside(i))

        # 11) Cria também um selector para centróides
        selector_cent = vtk.vtkSelectEnclosedPoints()
        selector_cent.SetInputData(input_poly_cent)
        selector_cent.SetSurfaceData(fib_surface)
        selector_cent.SetTolerance(1e-6)
        selector_cent.Update()

        inside_cent = np.zeros(n_tetra, dtype=bool)
        for i in range(n_tetra):
            inside_cent[i] = bool(selector_cent.IsInside(i))

        # 12) Reconstrói um array (n_tetra, 4) dizendo se cada vértice do tetra está dentro
        #     Cada tetra i corresponde aos índices flat_nodes[i*4 + 0..3]
        inside_nodes_per_tet = inside_unique_nodes[inverse_idx].reshape((n_tetra, 4))
        #      inside_nodes_per_tet[i,j] == True se o j-ésimo nó do tetra i estiver dentro.

        # 13) Agora a regra de decisão para cada tetra:
        #     Marca-se "fibrose" se (centróide dentro) OU (algum dos 4 nós dentro)
        tet_to_mark = (inside_cent) | (inside_nodes_per_tet.any(axis=1))

        # 14) Atualiza as tags
        tags_new[tet_to_mark] = 2

    # 15) Reescreve a mesh com as tags atualizadas
    new_cell_data = {}
    for key in cell_data:
        new_cell_data[key] = cell_data[key].copy()
        new_cell_data[key][tetra_index] = tags_new

    meshio.write(
        output_path,
        meshio.Mesh(
            points=points,
            cells=cells,
            cell_data=new_cell_data
        ),
        file_format="gmsh22",
        binary=False
    )

    # 16) Plot opcional dos centróides apenas para checar (não afeta marcação)
    if plot_centroids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=5, c='k')
        ax.set_title("Centróides dos Tetraedros")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marks fibrosis in the mesh using STL files.")
    parser.add_argument("--msh",      required=True, help="Path to the original .msh file")
    parser.add_argument("--stl_dir",  required=True, help="Directory containing the STL files")
    parser.add_argument("--output_path", required=True, help="Where to salvar o novo .msh com fibrose")
    parser.add_argument("--plot",     action="store_true", help="Se setado, plota os centróides")
    args = parser.parse_args()

    mark_fibrosis(args.msh, args.stl_dir, args.output_path, plot_centroids=args.plot)
