#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkSTLWriter.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // para std::stoi

int main(int argc, char *argv[])
{
    // 1) input.ply
    // 2) output.stl
    // 3) 0 ou 1 para indicar se é fibrose (flagScar)
    // Ex.: ./program input.ply output.stl 0
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " input.ply output.stl flagScar(0 or 1)" << std::endl;
        return EXIT_FAILURE;
    }

    // Argumentos
    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];
    bool flagScar = (std::stoi(argv[3]) != 0);

    // Leitura do arquivo PLY
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();

    // Guardar a malha original antes de suavizar
    vtkSmartPointer<vtkPolyData> originalMesh = reader->GetOutput();
    originalMesh->Register(nullptr); // garante que não seja desalocado prematuramente
    
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    // Suavização da malha
    if (!flagScar){
        smoother->SetInputData(originalMesh);
    }

    // Define o fator de suavização conforme flagScar
    if(flagScar)
        smoother->SetRelaxationFactor(0.0002);
    else
        smoother->SetRelaxationFactor(0.02);
    if(!flagScar){
        smoother->SetNumberOfIterations(200);
        smoother->BoundarySmoothingOff();
        smoother->BoundarySmoothingOff();
    }
    vtkSmartPointer<vtkPolyData> meshForNormals;
    
    if (!flagScar){
        smoother->Update();
        meshForNormals = smoother->GetOutput();
    }else
        meshForNormals = originalMesh;

    // Gera as normais a partir da malha adequada
    vtkSmartPointer<vtkPolyDataNormals> normals =
        vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInputData(meshForNormals);
    normals->FlipNormalsOn();
    normals->Update();
    normals->FlipNormalsOn();
    normals->Update();

    // Triangula a malha
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(normals->GetOutput());
    triangleFilter->Update();

    // Malha final suavizada
    vtkSmartPointer<vtkPolyData> finalMesh = triangleFilter->GetOutput();

    // --- Geração do mapeamento de vértices ---
    // Verificamos se o smoothing não mudou o número de vértices
    if (originalMesh->GetNumberOfPoints() == finalMesh->GetNumberOfPoints())
    {
        std::ofstream mapFile("vertex_mapping.txt");
        if (!mapFile.is_open())
        {
            std::cerr << "Could not open 'vertex_mapping.txt' for writing.\n";
            return EXIT_FAILURE;
        }

        vtkIdType numPoints = originalMesh->GetNumberOfPoints();
        for (vtkIdType i = 0; i < numPoints; i++)
        {
            double origPt[3];
            double smoothPt[3];

            // Pega ponto original
            originalMesh->GetPoint(i, origPt);
            // Pega ponto suavizado (mesmo índice i)
            finalMesh->GetPoint(i, smoothPt);

            // Formato: i origX origY origZ smoothX smoothY smoothZ
            mapFile << i << " "
                    << origPt[0] << " " << origPt[1] << " " << origPt[2] << " "
                    << smoothPt[0] << " " << smoothPt[1] << " " << smoothPt[2]
                    << "\n";
        }
        mapFile.close();
        // std::cout << "Vertex mapping saved to vertex_mapping.txt" << std::endl;
    }
    else
    {
        std::cerr << "[WARNING] Different number of points after smoothing. "
                  << "No vertex mapping generated.\n";
    }

    // Escreve a malha final em formato STL
    vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(outputFileName.c_str());
    writer->SetInputData(finalMesh);
    writer->SetFileTypeToASCII();
    writer->Write();

    // Libera a referência extra do originalMesh
    originalMesh->Delete();

    return EXIT_SUCCESS;
}
