import os
from datetime import datetime
import subprocess
import shutil

def execute_commands(patient_id):
    # Criar a pasta de saída com a data atual
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"saida/{date_str}/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Caminhos para superfícies e arquivos intermediários
    vtk_srf = f"{output_dir}/vtkFiles"
    msh_srf = f"{output_dir}/mshFiles"
    os.makedirs(vtk_srf, exist_ok=True)
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

        # Passo 4: Converter arquivos PLY para VTK
    ply_files = [
        f"./saida/plyFiles/{patient_id}-RVEpi.ply",
        f"./saida/plyFiles/{patient_id}-RVEndo.ply",
        f"./saida/plyFiles/{patient_id}-LVEpi.ply",
        f"./saida/plyFiles/{patient_id}-LVEndo.ply",
    ]

    vtk_outputs = [
        f"./saida/{date_str}/{patient_id}/vtkFiles/{patient_id}-RVEpi.vtk",
        f"./saida/{date_str}/{patient_id}/vtkFiles/{patient_id}-RVEndo.vtk",
        f"./saida/{date_str}/{patient_id}/vtkFiles/{patient_id}-LVEpi.vtk",
        f"./saida/{date_str}/{patient_id}/vtkFiles/{patient_id}-LVEndo.vtk",
    ]

    vtk_output_dir = f"./saida/{date_str}/{patient_id}/vtkFiles"
    os.makedirs(vtk_output_dir, exist_ok=True)

    for ply_file, vtk_output in zip(ply_files, vtk_outputs):
        if not os.path.exists(ply_file):
            print(f"Erro: Arquivo PLY {ply_file} não encontrado.")
            return
        try:
            ply_to_vtk_command = f"./convertPly2VTK/build/PlyToVtk {ply_file} {vtk_output}"
            subprocess.run(ply_to_vtk_command, shell=True, check=True)
            print(f"Arquivo VTK gerado com sucesso: {vtk_output}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao converter {ply_file} para {vtk_output}: {e}")
            return

   # Passo 5: Geração do arquivo `.msh` usando Gmsh

    lv_endo = f"{vtk_srf}/Patient_{patient_id}-LVEndo-Frame_1.vtk"
    rv_endo = f"{vtk_srf}/Patient_{patient_id}-RVEndo-Frame_1.vtk"
    rv_epi = f"{vtk_srf}/Patient_{patient_id}-RVEpi-Frame_1.vtk"
    scar_vtk = f"{vtk_srf}/Patient_{patient_id}_scar.vtk"
    msh = f"{msh_srf}/Patient_{patient_id}.msh"
    msh_srf_heart = f"{msh_srf}/Patient_{patient_id}_surf.msh"
    msh_heart = f"{msh_srf}/Patient_{patient_id}_model.msh"
    out_log = f"{msh_srf}/Patient_{patient_id}.out.txt"
    stl_scar = f"{output_dir}/stlFiles/Patient_scar.stl"

    try:
        # Gerar o arquivo .msh com superfícies do coração
        os.system('{} -3 {} -merge {} {} {} -o {} 2>&1 {}'.format(
            gmsh, lv_endo, rv_endo, rv_epi, biv_mesh_geo, msh, out_log))

        # Gerar o arquivo .msh das superfícies com merge de fibrose (quando aplicável)
        os.system('{} -3 {} -merge {} {} {} -o {}'.format(
            gmsh, lv_endo, rv_endo, rv_epi, biv_mesh_geo, msh_srf_heart))

        # Gerar o modelo final combinando superfícies e fibrose
        os.system('{} -3 {} -merge {} -o {}'.format(
            gmsh, msh_srf_heart, biv_msh_geo, msh_heart))

        # Limpar arquivos temporários (opcional)
        if os.path.exists(stl_scar):
            os.system('rm {}'.format(stl_scar))

        print(f"Modelo gerado com sucesso: {msh_heart}")

    except Exception as e:
        print(f"Erro ao gerar modelo: {e}")
        return

    print("Todos os comandos foram executados com sucesso!")


# Exemplo de uso
if __name__ == "__main__":
    patient_id = "Patient_1"
    execute_commands(patient_id)
