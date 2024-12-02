import open3d as o3d

def plot_ply(file_path):
    # Ler o arquivo .ply
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Verificar se a malha foi carregada corretamente
    if not mesh.is_empty():
        print("Arquivo .ply carregado com sucesso.")
        print(mesh)
        
        # Visualizar a malha
        o3d.visualization.draw_geometries([mesh])
    else:
        print("Erro ao carregar o arquivo .ply.")

# Exemplo de uso
plot_ply("Patient_1-RVEndo.ply")
