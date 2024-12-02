cmake_minimum_required(VERSION 3.10)
project(PlyToVtk)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(PlyToVtk ply_to_vtk.cpp)
target_link_libraries(PlyToVtk ${VTK_LIBRARIES})
