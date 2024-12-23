#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPLYReader.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " input.ply output.vtk" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];

    // Read PLY file
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();

    // Smooth the mesh
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoother = vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoother->SetInputData(reader->GetOutput());
    smoother->SetRelaxationFactor(0.02);
    smoother->SetNumberOfIterations(400);
    smoother->BoundarySmoothingOn();
    smoother->Update();

    // Generate surface normals
    vtkSmartPointer<vtkPolyDataNormals> normals = vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInputData(smoother->GetOutput());
    normals->FlipNormalsOn();
    normals->Update();

    // Write to VTK
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(outputFileName.c_str());
    writer->SetInputData(normals->GetOutput());
    writer->Write();

    std::cout << "File converted to: " << outputFileName << std::endl;

    return EXIT_SUCCESS;
}
