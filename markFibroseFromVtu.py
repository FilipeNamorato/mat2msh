import argparse
import os
import pandas as pd
import numpy as np
import vtk

def mark_fibrosis_in_alg(df, vtu_path, tolerance=1e-6, scale=1000):
    """
    Marca células no DataFrame .alg que estão dentro do volume VTU.
    """
    # 1) extrai centros
    centers = df[[0,1,2]].values

    # 2) lê o VTU
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    ugrid = reader.GetOutput()

    # 3) extrai a superfície (polydata) do VTU
    surfFilt = vtk.vtkDataSetSurfaceFilter()
    surfFilt.SetInputData(ugrid)
    surfFilt.Update()
    fib_surface = surfFilt.GetOutput()

    # 4) aplica escala (como você já fazia)
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    xfFilt = vtk.vtkTransformPolyDataFilter()
    xfFilt.SetTransform(transform)
    xfFilt.SetInputData(fib_surface)
    xfFilt.Update()
    fib_surface_scaled = xfFilt.GetOutput()

    # 5) configuro o SelectEnclosedPoints
    select = vtk.vtkSelectEnclosedPoints()
    select.SetSurfaceData(fib_surface_scaled)
    select.SetTolerance(tolerance)

    # 6) monto um PolyData com todos os centros
    pts = vtk.vtkPoints()
    for x,y,z in centers:
        pts.InsertNextPoint(x,y,z)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    select.SetInputData(pd)
    select.Update()

    # 7) tiro o resultado e atualizo a coluna 6 do df
    inside = [int(select.IsInside(i)) for i in range(len(centers))]
    df[6] = np.where(inside, 1, df[6])
    return sum(inside)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alg-in",  required=True,  help=".alg saudável")
    p.add_argument("--vtu-dir", required=True,  help="pasta com .vtu de fibrose")
    p.add_argument("--alg-out", required=True,  help=".alg de saída")
    p.add_argument("--tol",     type=float, default=0.5)
    p.add_argument("--scale",   type=float, default=1000.0)
    args = p.parse_args()

    df = pd.read_csv(args.alg_in, header=None)
    total = 0

    for fn in sorted(os.listdir(args.vtu_dir)):
        if fn.lower().endswith(".vtu"):
            path = os.path.join(args.vtu_dir, fn)
            marked = mark_fibrosis_in_alg(df, path, args.tol, args.scale)
            print(f"{fn}: {marked} volumes marcados")
            total += marked

    df.to_csv(args.alg_out, index=False, header=False)
    print(f"\nTotal marcado: {total}")
    print(f"Salvo em: {args.alg_out}")
