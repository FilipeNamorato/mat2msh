import os
import time
import argparse
import meshio
import numpy as np
import vtk
import matplotlib.pyplot as plt

def mark_fibrosis(path_msh, stl_dir, output_path="saida_com_fibrose.msh", plot_centroids=False):
    # Lê o .msh original
    mesh = meshio.read(path_msh)
    points = mesh.points
    cells = mesh.cells
    cell_data = mesh.cell_data

    # Localiza os tetraedros
    tetra_index = next(i for i, c in enumerate(cells) if c.type == "tetra")
    tetra = cells[tetra_index].data
    tags = cell_data["gmsh:physical"][tetra_index]

    # Calcula centróides dos tetraedros
    centroids = np.mean(points[tetra], axis=1)

    # Cria polydata com os centróides
    vtk_points = vtk.vtkPoints()
    for c in centroids:
        vtk_points.InsertNextPoint(c)
    input_poly = vtk.vtkPolyData()
    input_poly.SetPoints(vtk_points)

    # Itera sobre todos os STL no diretório e marca fibrose
    tags_new = tags.copy()
    for fname in sorted(os.listdir(stl_dir)):
        if not fname.endswith(".stl"):
            continue

        full_path = os.path.join(stl_dir, fname)

        # Espera adaptativa baseada em conteúdo
        for _ in range(20):  # tenta por até 2 segundos
            try:
                with open(full_path, "r", errors="ignore") as f:
                    lines = [next(f) for _ in range(5)]
                if any("vertex" in line for line in lines):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            print(f"[ERROR] STL {fname} is not ready or is malformed. Skipping.")
            continue

        print(f"Processing {fname}...")
        reader = vtk.vtkSTLReader()
        reader.SetFileName(full_path)
        reader.Update()
        fib_surface = reader.GetOutput()

        selector = vtk.vtkSelectEnclosedPoints()
        selector.SetInputData(input_poly)
        selector.SetSurfaceData(fib_surface)
        selector.SetTolerance(1e-6)
        selector.Update()

        is_inside = np.array([selector.IsInside(i) for i in range(len(centroids))])
        tags_new[is_inside == 1] = 2  # Marca como fibrose

    # Atualiza as tags na malha
    new_cell_data = {}
    for key in cell_data:
        new_cell_data[key] = cell_data[key].copy()
        new_cell_data[key][tetra_index] = tags_new

    # Salva a nova malha com marcações
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

    # Plota os centróides, se solicitado
    if plot_centroids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=5)
        ax.set_title("Centróides dos Tetraedros")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marks fibrosis in the mesh using STL files.")
    parser.add_argument("--msh", required=True, help="Path to the original .msh file")
    parser.add_argument("--stl_dir", required=True, help="Directory containing the STL files")
    parser.add_argument("--output_path", required=True, help="Path to save the .msh file with fibrosis")
    parser.add_argument("--plot", action="store_true", help="If set, plots the centroids")

    args = parser.parse_args()

    mark_fibrosis(args.msh, args.stl_dir, args.output_path, plot_centroids=args.plot)
