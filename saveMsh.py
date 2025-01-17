import numpy as np
import os
from datetime import datetime
from scipy.io import loadmat

def adjust_resolution(setstruct, structures, slice_thickness, slice_gap):
    """
    Adjusts the coordinates of the structures to account for spatial resolution (ResolutionX, ResolutionY).
    Calculates Z values based on SliceThickness and SliceGap.

    Parameters:
    - setstruct: Object containing the structures and resolution attributes.
    - structures: Dictionary with structures to be adjusted.
    - slice_thickness: Thickness between slices.
    - slice_gap: Gap between slices.

    Returns:
    - Adjusted setstruct with new scaled coordinates.
    - Indices of valid slices where at least one structure has valid data.
    """
    resolution_x = getattr(setstruct, 'ResolutionX', 1.0)
    resolution_y = getattr(setstruct, 'ResolutionY', 1.0)

    # Validate slices: at least one structure must be valid (not all NaNs)
    num_slices = None
    valid_slices_mask = None
    for name, (x_attr, y_attr) in structures.items():
        try:
            x_coords = getattr(setstruct, x_attr)
            y_coords = getattr(setstruct, y_attr)

            if x_coords.ndim == 3:
                num_points, _, num_slices = x_coords.shape
                valid_mask = ~np.isnan(x_coords[:, 0, :]) | ~np.isnan(y_coords[:, 0, :])
            elif x_coords.ndim == 2:
                num_points, num_slices = x_coords.shape
                valid_mask = ~np.isnan(x_coords) | ~np.isnan(y_coords)
            else:
                raise ValueError(f"Unexpected dimensions for {x_attr}: {x_coords.ndim}")

            # Check if at least one point is valid for each slice
            slice_validity = valid_mask.any(axis=0)

            # Combine masks: A slice is valid if at least one structure has valid points
            if valid_slices_mask is None:
                valid_slices_mask = slice_validity
            else:
                valid_slices_mask |= slice_validity

        except AttributeError:
            print(f"Error: {x_attr} or {y_attr} not found in the .mat file")
            return setstruct, None

    if valid_slices_mask is None or num_slices is None:
        print("No valid slices found.")
        return setstruct, None

    valid_indices = np.where(valid_slices_mask)[0]

    # Adjust coordinates for valid slices
    for name, (x_attr, y_attr) in structures.items():
        try:
            x_coords = getattr(setstruct, x_attr)
            y_coords = getattr(setstruct, y_attr)

            for s in valid_indices:
                if x_coords.ndim == 3:
                    x_coords[:, 0, s] = resolution_x * x_coords[:, 0, s]
                    y_coords[:, 0, s] = resolution_y * y_coords[:, 0, s]
                elif x_coords.ndim == 2:
                    x_coords[:, s] = resolution_x * x_coords[:, s]
                    y_coords[:, s] = resolution_y * y_coords[:, s]

            setattr(setstruct, x_attr, x_coords)
            setattr(setstruct, y_attr, y_coords)
        except AttributeError:
            print(f"Error: {x_attr} or {y_attr} not found in the .mat file")
        except ValueError as ve:
            print(f"ValueError: {ve}")

    return setstruct, valid_indices


def save_structures_to_txt(mat_filename, output_dir):
    """
    Saves the coordinates of structures (LVEndo, LVEpi, RVEndo, RVEpi) to unique .txt files,
    containing all slices for each structure in MATLAB format.

    Parameters:
    - mat_filename: Path to the .mat file.
    - output_dir: Directory where the files will be saved.
    """
    # Create an output directory based on the current date
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, date_str)
    os.makedirs(output_path, exist_ok=True)

    try:
        data = loadmat(mat_filename, struct_as_record=False, squeeze_me=True)
        setstruct = data['setstruct']
        slice_thickness = getattr(setstruct, 'SliceThickness', 8.0)
        slice_gap = getattr(setstruct, 'SliceGap', 0.0)
    except Exception as e:
        print(f"Error loading the .mat file: {e}")
        return None

    # Structures to be processed
    structures = {
        'LVEndo': ('EndoX', 'EndoY'),
        'LVEpi': ('EpiX', 'EpiY'),
        'RVEndo': ('RVEndoX', 'RVEndoY'),
        'RVEpi': ('RVEpiX', 'RVEpiY')
    }

    # Adjust resolutions and obtain valid slice indices
    print("Starting resolution adjustment...")
    setstruct, valid_indices = adjust_resolution(setstruct, structures, slice_thickness, slice_gap)
    print("Resolution adjustment completed.")

    if valid_indices is None:
        print("No valid slices to process.")
        return None

    # Process each structure
    for name, (x_attr, y_attr) in structures.items():
        try:
            x_coords = getattr(setstruct, x_attr)
            y_coords = getattr(setstruct, y_attr)

            num_points = x_coords.shape[0]

            # Create Z values starting at 0 for valid slices
            z_base = np.arange(len(valid_indices)) * (slice_thickness + slice_gap)
            z_values = np.tile(z_base, (num_points, 1))

            # Output file name
            output_filename = os.path.join(output_path, f"Patient_1-{name}.txt")
            with open(output_filename, 'w') as f:
                for relative_idx, s in enumerate(valid_indices):
                    # Get slice coordinates
                    x_slice = x_coords[:, 0, s] if x_coords.ndim == 3 else x_coords[:, s]
                    y_slice = y_coords[:, 0, s] if y_coords.ndim == 3 else y_coords[:, s]

                    # Filter valid points
                    valid_mask = ~np.isnan(x_slice) & ~np.isnan(y_slice)
                    valid_x = x_slice[valid_mask]
                    valid_y = y_slice[valid_mask]
                    valid_z = z_values[valid_mask, relative_idx]

                    # Write valid coordinates to the file
                    coords = np.column_stack((valid_x, valid_y, valid_z))
                    np.savetxt(f, coords, fmt="%.6f", delimiter=" ")

                print(f"File {output_filename} saved successfully.")

        except AttributeError:
            print(f"Error: {x_attr} or {y_attr} not found in the .mat file")

    print("Export completed successfully.")
    return output_path

def main():
    mat_filename = "./aligned_patient.mat"
    output_dir = "output"

    if not os.path.exists(mat_filename):
        print(f"Error: The file {mat_filename} does not exist.")
        return

    output_txt = save_structures_to_txt(mat_filename, output_dir)
    if not output_txt:
        print("Error during export to .txt.")
        return

    print(f"Files exported to directory: {output_txt}")

if __name__ == "__main__":
    main()
