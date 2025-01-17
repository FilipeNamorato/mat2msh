import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import LinearRegression

def read_mat(mat_filename, RVyes=False):
    print(f"Reading file: {mat_filename}")
    data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
    setstruct = data['setstruct']
    print("Data structure loaded successfully.")

    def get_coordinates(field):
        print(f"Extracting coordinates for: {field}")
        coords = getattr(setstruct, field, None)
        if coords is None:
            raise ValueError(f"Field {field} not found.")
        if len(coords.shape) == 2:
            coords = coords[:, np.newaxis, :]
        print(f"Coordinates for {field} extracted with shape: {coords.shape}")
        return coords

    # Get values of SliceThickness and SliceGap
    slice_thickness = getattr(setstruct, 'SliceThickness', None)
    slice_gap = getattr(setstruct, 'SliceGap', 0.0)  # Default to 0 if not present
    print(slice_gap)

    if slice_thickness is None:
        raise ValueError("Field 'SliceThickness' not found in the .mat file")

    # Calculate Z based on SliceThickness and SliceGap
    def calculate_z(num_slices, slice_thickness, slice_gap, num_points_per_slice):
        z_values = np.arange(num_slices) * slice_thickness
        return np.tile(z_values, (num_points_per_slice, 1))

    # Extract coordinates of structures
    endoX = get_coordinates('EndoX')
    endoY = get_coordinates('EndoY')
    epiX = get_coordinates('EpiX')
    epiY = get_coordinates('EpiY')

    num_slices = epiX.shape[2]
    num_points_per_slice = epiX.shape[0]

    # Generate Z values for all slices
    EndoZ = calculate_z(num_slices, slice_thickness, slice_gap, endoX.shape[0])
    EpiZ = calculate_z(num_slices, slice_thickness, slice_gap, epiX.shape[0])

    def calculate_barycenters(endoX, endoY, epiX, epiY):
        """Calculate barycenters for each slice, considering NaNs in EndoX."""
        barycenters = np.zeros((epiX.shape[2], 2))
        for s in range(epiX.shape[2]):
            if not np.isnan(endoX[0, 0, s]):  # If EndoX has no NaN for this slice
                barycenters[s, 0] = 0.5 * np.nanmean(epiX[:, :, s] + endoX[:, :, s])
                barycenters[s, 1] = 0.5 * np.nanmean(epiY[:, :, s] + endoY[:, :, s])
            else:
                barycenters[s, 0] = np.nanmean(epiX[:, :, s])
                barycenters[s, 1] = np.nanmean(epiY[:, :, s])
        return barycenters

    def align_slices(X, Y, barycenters, z_values):
        """Align slices based on linear regression of barycenters."""
        num_slices = X.shape[2]
        z = z_values.reshape(-1, 1)

        valid_idx = ~np.isnan(barycenters[:, 0])
        if valid_idx.sum() > 1:
            modelX = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 0])
            modelY = LinearRegression().fit(z[valid_idx], barycenters[valid_idx, 1])

            estX = modelX.predict(z)
            estY = modelY.predict(z)

            for s in range(num_slices):
                if not np.isnan(X[:, 0, s]).all():
                    shiftX = barycenters[s, 0] - estX[s]
                    shiftY = barycenters[s, 1] - estY[s]
                    X[:, :, s] -= shiftX
                    Y[:, :, s] -= shiftY

        return X, Y

    # Calculate barycenters and align Endo and Epi
    endo_barycenters = calculate_barycenters(endoX, endoY, epiX, epiY)
    endoX, endoY = align_slices(endoX, endoY, endo_barycenters, EndoZ[0])
    epiX, epiY = align_slices(epiX, epiY, endo_barycenters, EpiZ[0])

    if RVyes:
        # Align RV
        RVEndoX = get_coordinates('RVEndoX')
        RVEndoY = get_coordinates('RVEndoY')
        RVEpiX = get_coordinates('RVEpiX')
        RVEpiY = get_coordinates('RVEpiY')

        RVEndoZ = calculate_z(num_slices, slice_thickness, slice_gap, RVEndoX.shape[0])
        RVEpiZ = calculate_z(num_slices, slice_thickness, slice_gap, RVEpiX.shape[0])

        rv_barycenters = calculate_barycenters(RVEndoX, RVEndoY, RVEpiX, RVEpiY)

        # Align RV Endo and RV Epi slices
        RVEndoX, RVEndoY = align_slices(RVEndoX, RVEndoY, rv_barycenters, RVEndoZ[0])
        RVEpiX, RVEpiY = align_slices(RVEpiX, RVEpiY, rv_barycenters, RVEpiZ[0])

        # Update setstruct with aligned RV coordinates
        setstruct.RVEndoX = RVEndoX
        setstruct.RVEndoY = RVEndoY
        setstruct.RVEpiX = RVEpiX
        setstruct.RVEpiY = RVEpiY

    print("Alignment complete.")

    # Update the original structure with aligned coordinates
    setstruct.EndoX = endoX
    setstruct.EndoY = endoY
    setstruct.EpiX = epiX
    setstruct.EpiY = epiY

    # Save the processed file
    output_filename = 'aligned_patient.mat'
    savemat(output_filename, {'setstruct': setstruct}, do_compression=True)
    print(f"Aligned file saved as: {output_filename}")
