# Example usage:
# python readScar.py ./Patient_1_new.mat --shiftx endo_shifts_x.txt --shifty endo_shifts_y.txt
import os
import sys
import glob
import subprocess
import argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Simple ROI entry: name, slice z, and list of (x,y) points
ROIEntry = namedtuple('ROIEntry', ['name', 'z', 'points'])

def readScar(mat_filename):
    """
    Read .mat file and return a flat list of ROIEntry(name, z, points).
    Associates each slice with its correct ROI name.
    """
    print(f"Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename)
    rois = data['setstruct'][0][0]['Roi']

    entries = []
    for idx, roi in enumerate(rois):
        # extract per-slice names
        raw_names = roi['Name'].flatten()
        slice_names = []
        for element in raw_names:
            # flatten to string
            if isinstance(element, np.ndarray):
                val = element.flat[0]
            else:
                val = element
            slice_names.append(str(val).strip())
        # coordinate arrays
        X = roi['X']
        Y = roi['Y']
        Z = roi['Z']
        for i in range(len(Z)):
            z_val = int(np.atleast_1d(Z[i])[0])
            x_arr = np.atleast_1d(X[i]).flatten()
            y_arr = np.atleast_1d(Y[i]).flatten()
            if x_arr.size == 0 or y_arr.size == 0:
                continue
            name = slice_names[i] if i < len(slice_names) else slice_names[0]
            pts = list(zip(x_arr, y_arr))
            entries.append(ROIEntry(name, z_val, pts))
    return entries


def group_by_slice(entries):
    """Group ROIEntry list into dict of slice-> {name: points}."""
    fatias_txt = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias_txt[e.z][e.name].extend(e.points)
    return fatias_txt


def plot_slices(fatias):
    for z, roi_map in sorted(fatias.items()):
        plt.figure(figsize=(6,6))
        for name, pts in roi_map.items():
            arr = np.array(pts)
            plt.scatter(arr[:,0], arr[:,1], label=name, s=20)
            cen = arr.mean(axis=0)
            plt.text(cen[0], cen[1], name,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', alpha=0.5))
        plt.title(f"Slice Z={z}")
        plt.gca().invert_yaxis()
        plt.legend(fontsize=7)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir="fatias"):
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)
    for z, roi_map in fatias.items():
        sx = shifts_x[z] if 0 <= z < len(shifts_x) else 0
        sy = shifts_y[z] if 0 <= z < len(shifts_y) else 0
        fname = os.path.join(output_dir, f"fatia_{z}.txt")
        with open(fname,'w') as f:
            for pts in roi_map.values():
                for x,y in pts:
                    f.write(f"{x-sx} {y-sy} {z}\n")
        print(f"Saved slice {z} to {fname}")


def main():
    parser = argparse.ArgumentParser(description="ROI extraction pipeline")
    parser.add_argument('matfile', help='Path to .mat file')
    parser.add_argument('--shiftx', default='endo_shifts_x.txt', help='Shift X file')
    parser.add_argument('--shifty', default='endo_shifts_y.txt', help='Shift Y file')
    args = parser.parse_args()

    entries = readScar(args.matfile)
    fatias = group_by_slice(entries)
    plot_slices(fatias)
    save_fatias_to_txt(fatias, args.shiftx, args.shifty)

if __name__ == '__main__':
    main()
