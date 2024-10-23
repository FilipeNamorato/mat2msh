import sys
from scipy.io import savemat
from readMat import read_mat

def main():
    if len(sys.argv) < 3 or sys.argv[1] != '-m':
        print("Uso: python3 main.py -m <nome_do_arquivo>")
        sys.exit(1)

    mat_filename = sys.argv[2] + ".mat"
    print(f"Lendo e processando o arquivo: {mat_filename}")

    data = read_mat(mat_filename)
    output_filename = "analise.mat"

    print(f"Salvando o arquivo alinhado como: {output_filename}")
    savemat(output_filename, data)

    print("Processamento concluído com sucesso.")

if __name__ == "__main__":
    main()
