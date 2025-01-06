import os
from datetime import datetime
import subprocess
import shutil

def execute_commands(patient_id):
    # Criar a pasta de saída com a data atual
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"./saida/{date_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Comando 1: Executar main.py
    try:
        main_command = f"python3 main.py -m {patient_id}"
        subprocess.run(main_command, shell=True, check=True)
        print("Finalizado processo de alinhamento")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar main.py: {e}")
        return

    # Comando 2: Executar saveMsh.py
    try:
        save_msh_command = "python3 saveMsh.py"
        subprocess.run(save_msh_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar saveMsh.py: {e}")
        return

    # Comando 3: Gerar superfícies
    surface_files = [
        f"{output_dir}/{patient_id}-LVEndo.txt",
        f"{output_dir}/{patient_id}-LVEpi.txt",
        f"{output_dir}/{patient_id}-RVEndo.txt",
        f"{output_dir}/{patient_id}-RVEpi.txt",
    ]

    for surface_file in surface_files:
        try:
            surface_command = f"python3 make_surface.py {surface_file}"
            subprocess.run(surface_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao gerar superfície para {surface_file}: {e}")
            return

    # Comando 4: Converter arquivos PLY para VTK
    ply_files = [
        f"./saida/plyFiles/{patient_id}-RVEpi.ply",
        f"./saida/plyFiles/{patient_id}-RVEndo.ply",
        f"./saida/plyFiles/{patient_id}-LVEpi.ply",
        f"./saida/plyFiles/{patient_id}-LVEndo.ply",
    ]

    vtk_outputs = [
        f"./convertPly2VTK/result/outputRVEpi.vtk",
        f"./convertPly2VTK/result/outputRVEndo.vtk",
        f"./convertPly2VTK/result/outputLVEpi.vtk",
        f"./convertPly2VTK/result/outputLVEndo.vtk",
    ]

    for ply_file, vtk_output in zip(ply_files, vtk_outputs):
        try:
            ply_to_vtk_command = f"./convertPly2VTK/build/PlyToVtk {ply_file} {vtk_output}"
            subprocess.run(ply_to_vtk_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao converter {ply_file} para {vtk_output}: {e}")
            return

    # Comando 5: Gerar arquivo .msh usando os arquivos .vtk
    gmsh = "/home/filipe/.local/bin/gmsh"
    if not os.path.exists(gmsh):
        print("Erro: Gmsh não encontrado no caminho especificado.")
        return
    print("teste")
    
    vtk_inputs = [
        vtk_outputs[0],  # outputRVEpi.vtk
        vtk_outputs[1],  # outputRVEndo.vtk
        vtk_outputs[2],  # outputLVEpi.vtk
        vtk_outputs[3],  # outputLVEndo.vtk
    ]

    
    msh_file = f"{output_dir}/{patient_id}.msh"
    geo_script = "./scripts/biv_mesh.geo"

    
    # Verificar se todos os arquivos existem
    required_files = vtk_inputs + [geo_script]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Erro: Arquivo necessário não encontrado - {file_path}")
            return

    # Comando para rodar o Gmsh
    gmsh_command = [
        gmsh,
        "-3",
        vtk_inputs[3],  # LVEndo como entrada principal
        "-merge", vtk_inputs[0], vtk_inputs[1], vtk_inputs[2], geo_script,
        "-o", msh_file,
    ]

    try:
        result = subprocess.run(gmsh_command, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Erro ao executar o Gmsh para {patient_id}. Detalhes: {result.stderr}")
            return
        print(f"Arquivo .msh gerado com sucesso para {patient_id}: {msh_file}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o Gmsh: {e}")
        return

    print("Teste")
    exit(0)
    print("Todos os comandos foram executados com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    patient_id = "Patient_1"
    execute_commands(patient_id)
