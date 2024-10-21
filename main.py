import argparse
import scipy.io as sci
from readMat import read_mat  # Importa a função de leitura
from scipy.io import savemat

def main():
    # Configura o argparse para aceitar o nome do arquivo .mat como argumento
    parser = argparse.ArgumentParser(description='Processamento de arquivos MATLAB.')
    parser.add_argument('-m', type=str, required=True, help='Nome do arquivo MATLAB (.mat)')

    args = parser.parse_args()
    mat_filename = args.m + ".mat"

    # Carrega o arquivo original .mat
    data = sci.loadmat(mat_filename)

    # Chama a função de leitura e processamento para obter as alterações
    mat_shift = read_mat(mat_filename)

    # Atualiza os campos dentro de setstruct com as alterações
    setstruct = data['setstruct'][0][0]
    for key, value in mat_shift.items():
        if key in setstruct.dtype.names:
            setstruct[key] = value

    # Define o nome do arquivo de saída
    output_filename = "Teste1.mat"  # Ou você pode usar `mat_filename` para sobrescrever

    # Salva o arquivo atualizado, preservando os dados não alterados
    savemat(output_filename, data)

    print(f"Arquivo salvo como: {output_filename}")

if __name__ == "__main__":
    main()
