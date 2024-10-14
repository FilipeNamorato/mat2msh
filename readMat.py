import gmsh
import numpy as np
import scipy.io as sci



def read_mat(mat):
    data = sci.loadmat(mat)
    