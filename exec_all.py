import os
from datetime import datetime
import subprocess

def execute_commands(patient_id):
    # Criar a pasta de saída com a data atual
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"saida/{date_str}/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Caminhos para superfícies e arquivos intermediários
    stl_srf = f"{output_dir}/stlFiles"
    msh_srf = f"{output_dir}/mshFiles"
    os.makedirs(stl_srf, exist_ok=True)
    os.makedirs(msh_srf, exist_ok=True)

    # Gmsh e scripts de geração
    gmsh = "/home/filipe/.local/bin/gmsh"
    biv_mesh_geo = "./scripts/biv_mesh.geo"
    biv_msh_geo = "./scripts/biv_msh.geo"

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
        f"./saida/{date_str}/{patient_id}-LVEndo.txt",
        f"./saida/{date_str}/{patient_id}-LVEpi.txt",
        f"./saida/{date_str}/{patient_id}-RVEndo.txt",
        f"./saida/{date_str}/{patient_id}-RVEpi.txt",
    ]

    for surface_file in surface_files:
        try:
            surface_command = f"python3 make_surface.py {surface_file}"
            subprocess.run(surface_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao gerar superfície para {surface_file}: {e}")
            return

    # Passo 4: Converter arquivos PLY para STL
    ply_files = [
        f"./saida/plyFiles/{patient_id}-RVEpi.ply",
        f"./saida/plyFiles/{patient_id}-RVEndo.ply",
        f"./saida/plyFiles/{patient_id}-LVEpi.ply",
        f"./saida/plyFiles/{patient_id}-LVEndo.ply",
    ]

    stl_outputs = [
        f"./saida/{date_str}/{patient_id}/stlFiles/{patient_id}-RVEpi.stl",
        f"./saida/{date_str}/{patient_id}/stlFiles/{patient_id}-RVEndo.stl",
        f"./saida/{date_str}/{patient_id}/stlFiles/{patient_id}-LVEpi.stl",
        f"./saida/{date_str}/{patient_id}/stlFiles/{patient_id}-LVEndo.stl",
    ]

    for ply_file, stl_output in zip(ply_files, stl_outputs):
        if not os.path.exists(ply_file):
            print(f"Erro: Arquivo PLY {ply_file} não encontrado.")
            return
        try:
            ply_to_stl_command = f"./convertPly2STL/build/PlyToStl {ply_file} {stl_output}"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            print(f"Arquivo STL gerado com sucesso: {stl_output}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao converter {ply_file} para {stl_output}: {e}")
            return

    # Código antes de `exit(0)`
    print("### Processamento Final ###")
    print(f"Arquivos STL gerados com sucesso em: {stl_srf}")
    print("Finalizando o script antes de executar o Gmsh e outras etapas.")

    exit(0)

    # Passo 5: Geração do arquivo `.msh` usando Gmsh
    lv_endo = f"{stl_srf}/{patient_id}-LVEndo.stl"
    rv_endo = f"{stl_srf}/{patient_id}-RVEndo.stl"
    rv_epi = f"{stl_srf}/{patient_id}-RVEpi.stl"
    msh_heart = f"{msh_srf}/Patient_{patient_id}_model.msh"

    try:
        gmsh_command = f"{gmsh} -3 -merge {lv_endo} {rv_endo} {rv_epi} -o {msh_heart}"
        subprocess.run(gmsh_command, shell=True, check=True)
        print(f"Modelo gerado com sucesso: {msh_heart}")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao gerar modelo: {e}")
        return

    print("Todos os comandos foram executados com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    patient_id = "Patient_1"
    execute_commands(patient_id)
