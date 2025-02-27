#!/usr/bin/env python3
import vtk
import sys

def merge_stls(input_files, output_file, do_smooth=False, relaxation=0.02, iterations=200):

    # 1) Cria um objeto para "somar" (append) vários PolyData
    append_filter = vtk.vtkAppendPolyData()

    # 2) Lê cada STL e adiciona no append_filter
    for stl_path in input_files:
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_path)
        reader.Update()
        polyData = reader.GetOutput()

        # Se a malha estiver vazia ou com erro, evite problemas
        if not polyData or polyData.GetNumberOfPoints() == 0:
            print(f"Aviso: '{stl_path}' está vazio ou inválido. Ignorando.")
            continue

        append_filter.AddInputData(polyData)

    # 3) Faz o merge
    append_filter.Update()
    merged_polydata = append_filter.GetOutput()

    print(f"Merged {len(input_files)} arquivos em um só. Total de {merged_polydata.GetNumberOfPoints()} pontos.")

    # 4) (Opcional) Aplica suavização na malha unificada
    if do_smooth:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(merged_polydata)
        smoother.SetRelaxationFactor(relaxation)
        smoother.SetNumberOfIterations(iterations)
        smoother.BoundarySmoothingOff()  # Ou .On() se quiser suavizar bordas livres
        smoother.Update()
        merged_polydata = smoother.GetOutput()
        print(f"Suavização aplicada (relax={relaxation}, iterações={iterations}).")

    # 5) Salva o resultado em STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(merged_polydata)
    writer.SetFileTypeToASCII()  # ou binário, se preferir: .SetFileTypeToBinary()
    writer.Write()

    print(f"Arquivo final salvo em: {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Uso: python merge_stls.py output.stl [--smooth] arquivo1.stl arquivo2.stl ...")
        sys.exit(1)

    # 1) Primeiro argumento = arquivo de saída
    output_file = sys.argv[1]

    # 2) Verifica se '--smooth' está presente
    do_smooth = False
    input_files = sys.argv[2:]
    if '--smooth' in input_files:
        do_smooth = True
        input_files.remove('--smooth')

    # 3) Chama a função de merge
    merge_stls(input_files, output_file, do_smooth=do_smooth)


if __name__ == "__main__":
    main()
