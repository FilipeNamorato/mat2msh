import numpy as np
from scipy.io import loadmat

def readScar():
    mat_filename = "PatientData.mat"
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']

    print("Data structure loaded successfully.")

    # Extrair dados do ROI
    def extract_roi_data(setstruct):
        """
        Extrai dados do ROI incluindo coordenadas e metadados.

        :param setstruct: Estrutura carregada do .mat contendo o ROI.
        :return: Lista de registros do ROI com campos relevantes.
        """
        roi_data = getattr(setstruct, 'Roi', None)
        if roi_data is None:
            raise ValueError("'Roi' field not found in setstruct.")

        records = []

        # Verificar se roi_data possui múltiplos registros ou um único
        if isinstance(roi_data, (list, np.ndarray)):
            for record in roi_data:
                x_coords = getattr(record, 'X', None)  # Campo X
                y_coords = getattr(record, 'Y', None)  # Campo Y
                z_value = getattr(record, 'Z', None)  # Exemplo de outro campo
                roi_name = getattr(record, 'Name', None)  # Nome do ROI

                records.append({
                    'X': x_coords,
                    'Y': y_coords,
                    'Z': z_value,
                    'Name': roi_name
                })
        else:  # Caso seja um único objeto
            x_coords = getattr(roi_data, 'X', None)  # Campo X
            y_coords = getattr(roi_data, 'Y', None)  # Campo Y
            z_value = getattr(roi_data, 'Z', None)  # Exemplo de outro campo
            roi_name = getattr(roi_data, 'Name', None)  # Nome do ROI

            records.append({
                'X': x_coords,
                'Y': y_coords,
                'Z': z_value,
                'Name': roi_name
            })

        return records

    # Obter dados do ROI
    roi_records = extract_roi_data(setstruct)

    # Exemplo de iteração sobre os registros extraídos
    for i, roi in enumerate(roi_records):
        x_shape = roi['X'].shape if roi['X'] is not None else "None"
        y_shape = roi['Y'].shape if roi['Y'] is not None else "None"
        print(f"ROI {i}: Name={roi['Name']}, X Shape={x_shape}, Y Shape={y_shape}")

    return roi_records

if __name__ == "__main__":
    roi_records = readScar()
