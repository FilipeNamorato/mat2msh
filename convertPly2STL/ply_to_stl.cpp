#include <vtkSmartPointer.h>
#include <vtkPLYReader.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkTriangleFilter.h>
#include <vtkSTLWriter.h>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " input.ply output.stl" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];

    // Leitura do arquivo PLY
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();

    // Suavização da malha
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoother->SetInputData(reader->GetOutput());
    smoother->SetRelaxationFactor(0.02);
    smoother->SetNumberOfIterations(400);
    smoother->BoundarySmoothingOn();
    smoother->Update();

    // Geração de normais na superfície
    vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInputData(smoother->GetOutput());
    normals->FlipNormalsOn();
    normals->Update();

    // Triangulação da malha
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
    triangleFilter->SetInputData(normals->GetOutput());
    triangleFilter->Update();

    // Escrita do arquivo STL
    vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(outputFileName.c_str());
    writer->SetInputData(triangleFilter->GetOutput());
    writer->SetFileTypeToBinary(); // Salvar como binário para melhor compatibilidade
    writer->Write();

    std::cout << "Arquivo STL gerado com sucesso: " << outputFileName << std::endl;

    return EXIT_SUCCESS;
}
