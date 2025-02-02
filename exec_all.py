import os
from datetime import datetime
import subprocess
from readMat import read_mat
import argparse

def execute_commands(input_file):
    # Fixed patient ID for processing
    patient_id = "Patient_1"

    # Create the output folder with the current date
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"output/{date_str}/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Paths for surfaces and intermediate files
    stl_srf = f"{output_dir}/stlFiles"
    msh_srf = f"{output_dir}/mshFiles"
    os.makedirs(stl_srf, exist_ok=True)
    os.makedirs(msh_srf, exist_ok=True)

    # Gmsh and generation scripts
    gmsh = "./scripts/gmsh-2.13.1/bin/gmsh"
    biv_mesh_geo = "./scripts/biv_mesh.geo"
    biv_msh_geo = "./scripts/biv_msh.geo"

    print("========================================================================================")
    # Step 1: Process the .mat file
    print(f"Processing the file: {input_file}")

    try:
        # Attempt to process the .mat file
        read_mat(input_file)
        print("MAT file processed successfully.")
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file {input_file} does not exist.")
        return
    except ValueError as ve:
        # Handle cases where the file content is invalid
        print(f"Error: Invalid content in {input_file}: {ve}")
        return
    except Exception as e:
        # Handle any other unforeseen errors
        print(f"Unexpected error while processing {input_file}: {e}")
        return
    print("========================================================================================")
    # Step 2: Execute saveMsh.py
    try:
        save_msh_command = "python3 saveMsh.py"
        subprocess.run(save_msh_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing saveMsh.py: {e}")
        return
    print("========================================================================================")

    # Step 3: Generate surfaces
    surface_files = [
        f"./output/{date_str}/{patient_id}-LVEndo.txt",
        f"./output/{date_str}/{patient_id}-LVEpi.txt",
        f"./output/{date_str}/{patient_id}-RVEndo.txt",
        f"./output/{date_str}/{patient_id}-RVEpi.txt",
    ]

    for surface_file in surface_files:
        try:
            surface_command = f"python3 make_surface.py {surface_file}"
            subprocess.run(surface_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating surface for {surface_file}: {e}")
            return
    print("========================================================================================")

    # Step 4: Convert PLY files to STL
    ply_files = [
        f"./output/plyFiles/{patient_id}-RVEpi.ply",
        f"./output/plyFiles/{patient_id}-RVEndo.ply",
        f"./output/plyFiles/{patient_id}-LVEpi.ply",
        f"./output/plyFiles/{patient_id}-LVEndo.ply",
    ]

    stl_outputs = [
        f"./output/{date_str}/{patient_id}/stlFiles/{patient_id}-RVEpi.stl",
        f"./output/{date_str}/{patient_id}/stlFiles/{patient_id}-RVEndo.stl",
        f"./output/{date_str}/{patient_id}/stlFiles/{patient_id}-LVEpi.stl",
        f"./output/{date_str}/{patient_id}/stlFiles/{patient_id}-LVEndo.stl",
    ]

    for ply_file, stl_output in zip(ply_files, stl_outputs):
        if not os.path.exists(ply_file):
            print(f"Error: PLY file {ply_file} not found.")
            return
        try:
            ply_to_stl_command = f"./convertPly2STL/build/PlyToStl {ply_file} {stl_output}"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            print(f"STL file generated successfully: {stl_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {ply_file} to {stl_output}: {e}")
            return

    # Final Processing
    print("### Final Processing ###")
    print(f"STL files generated successfully in: {stl_srf}")
    print("Ending the script before executing Gmsh and other steps.")

    print("========================================================================================")
    # Step 5: Generate the `.msh` file using Gmsh
    lv_endo = f"{stl_srf}/{patient_id}-LVEndo.stl"
    rv_endo = f"{stl_srf}/{patient_id}-RVEndo.stl"
    rv_epi = f"{stl_srf}/{patient_id}-RVEpi.stl"
    msh_heart = f"{msh_srf}/{patient_id}.msh"
    out_log = f"{msh_srf}/{patient_id}.log"

    # Command with os.system
    try:
        os.system('{} -3 {} -merge {} {} {} -o {} 2>&1 {}'.format(
            gmsh, lv_endo, rv_endo, rv_epi, biv_mesh_geo, msh_heart, out_log))
        print(f"Model generated successfully: {msh_heart}")
    except Exception as e:
        print(f"Error generating model: {e}")
        return
    print("========================================================================================")
    print("Finished processing the patient data.")
    print("========================================================================================")
    os.remove("./aligned_patient.mat")
    
# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute pipeline for processing a .mat file.")
    parser.add_argument("-i", "--input_file", required=True, help="Full path to the input .mat file")

    args = parser.parse_args()

    execute_commands(args.input_file)
