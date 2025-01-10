cmake_minimum_required(VERSION 3.10)
project(PlyToStl)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(PlyToStl ply_to_stl.cpp)
target_link_libraries(PlyToStl ${VTK_LIBRARIES})
