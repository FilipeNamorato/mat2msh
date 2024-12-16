import os
from datetime import datetime
import subprocess

def execute_commands(patient_id):
    # Criar a pasta de saída com a data atual
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"./saida/{date_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Comando 1: Executar main.py
    main_command = f"python3 main.py -m {patient_id}"
    subprocess.run(main_command, shell=True, check=True)
    
    # Comando 2: Executar saveMsh.py
    save_msh_command = "python3 saveMsh.py"
    subprocess.run(save_msh_command, shell=True, check=True)
    
    # Comando 3: Gerar superfícies
    surface_files = [
        f"{output_dir}/{patient_id}-LVEndo.txt",
        f"{output_dir}/{patient_id}-LVEpi.txt",
        f"{output_dir}/{patient_id}-RVEndo.txt",
        f"{output_dir}/{patient_id}-RVEpi.txt",
    ]

    for surface_file in surface_files:
        surface_command = f"python3 make_surface.py {surface_file}"
        subprocess.run(surface_command, shell=True, check=True)
    
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
        ply_to_vtk_command = f"./convertPly2VTK/build/PlyToVtk {ply_file} {vtk_output}"
        subprocess.run(ply_to_vtk_command, shell=True, check=True)
    
    print("Todos os comandos foram executados com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    patient_id = "Patient_1"
    execute_commands(patient_id)
