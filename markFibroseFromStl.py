import argparse
import os
import pandas as pd
import numpy as np
import vtk
#python markFibroseFromStl.py --alg-in ../arquivosTesteAlgSTL/outputTesteAlg.alg --stl-dir ../arquivosTesteAlgSTL/fibrose/ --alg-out mesh_com_fib.alg --tol 0.5 --scale 1000
def mark_fibrosis_in_alg(df, stl_path, tolerance=1e-6, scale=1000):
    """
    Mark cells in the .alg DataFrame that are inside the given STL surface.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the .alg mesh data.
        stl_path (str): Path to the STL file.
        tolerance (float): Tolerance for vtkSelectEnclosedPoints.
        scale (float): Scale factor applied to STL geometry (usually matches -r used in HexaMeshFromVTK).

    Returns:
        int: Number of marked volumes (cells).
    """
    # Extrai os centros das células (colunas 0, 1 e 2)
    centers = df[[0, 1, 2]].values

    # Lê o arquivo STL da fibrose
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    fib_surface = reader.GetOutput()

    # Aplica fator de escala no STL (usado para alinhar com o .alg gerado com -r 1000)
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetTransform(transform)
    filter.SetInputData(fib_surface)
    filter.Update()
    fib_surface_scaled = filter.GetOutput()

    # Prepara o filtro para identificar quais pontos estão dentro da superfície STL
    select_enclosed = vtk.vtkSelectEnclosedPoints()
    select_enclosed.SetSurfaceData(fib_surface_scaled)
    select_enclosed.SetTolerance(tolerance)

    # Cria uma estrutura com todos os centros da malha para verificar
    points = vtk.vtkPoints()
    for x, y, z in centers:
        points.InsertNextPoint(x, y, z)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Informa ao VTK quais pontos devem ser testados: neste caso, os centros dos volumes da malha .alg
    select_enclosed.SetInputData(polydata)

    # Executa o algoritmo que verifica, para cada ponto, se ele está dentro da superfície STL
    select_enclosed.Update()

    # Inicializa uma lista para armazenar os índices dos volumes marcados como fibrose
    inside = []

    # Itera sobre todos os centros das células
    for i in range(len(centers)):
        is_inside = select_enclosed.IsInside(i)  # Verifica se o ponto está dentro da superfície STL
        inside.append(int(is_inside))            # Adiciona 1 (dentro) ou 0 (fora) à lista

    # Atualiza a coluna 6 ('tecido') apenas onde foi identificado fibrose
    df[6] = np.where(inside, 1, df[6])

    return sum(inside)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark fibrosis in a healthy .alg file using multiple STL surfaces")
    parser.add_argument("--alg-in", required=True, help="Path to the healthy .alg file")
    parser.add_argument("--stl-dir", required=True, help="Directory containing STL files for fibrotic regions")
    parser.add_argument("--alg-out", required=True, help="Output .alg file with fibrosis marked")
    parser.add_argument("--tol", type=float, default=0.5, help="Tolerance used for the inside check")
    parser.add_argument("--scale", type=float, default=1000.0, help="Scale factor for STL (match with -r used in mesh generation)")
    args = parser.parse_args()

    # Carrega o .alg saudável
    df = pd.read_csv(args.alg_in, header=None)
    total_marked = 0

    # Itera sobre todos os arquivos .stl no diretório informado
    for fname in sorted(os.listdir(args.stl_dir)):
        if fname.endswith('.stl'):
            stl_path = os.path.join(args.stl_dir, fname)
            marked = mark_fibrosis_in_alg(df, stl_path, args.tol, args.scale)
            print(f"{fname}: {marked} volumes marked as fibrosis")
            total_marked += marked

    # Salva o novo .alg com as marcações acumuladas
    df.to_csv(args.alg_out, index=False, header=False)
    print(f"\nTotal marked volumes: {total_marked}")
    print(f"Output saved to: {args.alg_out}")
