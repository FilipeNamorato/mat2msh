Mesh.CharacteristicLengthMin = 1;
Mesh.CharacteristicLengthMax = 3;

Coherence Mesh;

Merge "../esfera.stl";

// Verificar se superfícies existem antes de definir os volumes
If (Exists(Surface {7}) && Exists(Surface {8}) && Exists(Surface {9}))
    Surface Loop(1) = {7, 8, 9, 11, 12};
    Volume(1) = {1};
    Physical Volume("healthy", 1) = {1};
    
    Surface Loop(2) = {12};
    Volume(2) = {2};
    Physical Volume("fibrose", 2) = {2};
Else
    Info("Algumas superfícies estão ausentes. Criando apenas o volume saudável.");
    Surface Loop(1) = {7, 8, 9, 11};
    Volume(1) = {1};
    Physical Volume("healthy", 1) = {1};
EndIf

Coherence;