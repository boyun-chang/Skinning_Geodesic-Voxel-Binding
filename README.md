# Skinning_Geodesic-Voxel-Binding
CUDA must be set up to run the program, and MAYA is required to view the results.

Development Process
1. Extracting skeleton hierarchy data with a script in Maya.
2. Run this code(https://github.com/sueda/fbx-extract) to extract skeleton transform data.
3. Make window for view debug.
4. Voxelize a Character Mesh.
5. Tag Voxel with the scanline algorithm. (interior node, boundary node, empty node)
6. Tag Skeleton node on Voxel with skeleton data.
7. Finding the shortest distance from skeleton node to boundary node using Dijkstra's algorithm
8. Calculate the distance from the voxel center to the mesh vertex, then add it to the geodesic distance to obtain the weight for each vertex.
9. Export result for csv file.
10. Change csv to xml by a script in Maya.
11. Apply weights to the character mesh with Deform > Import Weights and check the results.

Result

<img width="521" height="520" alt="Image" src="https://github.com/user-attachments/assets/878b4840-b142-4798-9d40-e66ae679a4b9" />
![Image](https://github.com/user-attachments/assets/8a00ab65-7f8b-4c0b-9a83-8570c1764747)



