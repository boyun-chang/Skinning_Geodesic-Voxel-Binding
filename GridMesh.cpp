#include "GridMesh.h"
#include "Mesh.h"
#include "GL\glut.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>

void Grid::id(int i, int j, int k)
{
	_i = i;
	_j = j;
	_k = k;
}

void Grid::draw(void)
{
	glPushMatrix();
	auto size = _max - _min;
	auto center = (_max + _min) / 2.0;
	glTranslatef(center.x(), center.y(), center.z());
	glutWireCube(size.x());	
	glPopMatrix();
}

void Grid::drawPoint(void)
{
	glPushMatrix();
	auto size = _max - _min;
	auto center = (_max + _min) / 2.0;
	glPointSize(10.0f);
	glBegin(GL_POINTS);
	glVertex3f(center.x(), center.y(), center.z());
	glEnd();
	glPointSize(1.0f);

	glPopMatrix();
}
/*
Edge* Grid::Connect(Grid *grid, double weight)
{
	Edge* e = new Edge(this, grid, weight);
	_edges.push_back(e);
	grid->_edges.push_back(e);
	return e;
}*/

GridMesh::GridMesh(Mesh *mesh, int res)
{
	_res = res;
	//_hashRes = (int)(1.0 / 0.01);
	_hashRes = _res;
	_mesh = mesh;
	_min.Set(0.0);
	_max.Set(1.0);

	_hashTable = new HashTable(_hashRes);
	init();
	
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	scanline(); // 계산
	std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
	printf("%f sec\n", sec.count());
}

GridMesh::~GridMesh(void)
{
}

bool GridMesh::intersect(Face *face, Grid *grid)
{
	Vec3<double> boxNormals[3];
	boxNormals[0].Set(1, 0, 0);
	boxNormals[1].Set(0, 1, 0);
	boxNormals[2].Set(0, 0, 1);

	// Test the box normals
	for (int i = 0; i < 3; i++) {
		auto boxVolume = project(face, boxNormals[i]);		
		if (boxVolume.second < grid->_min[i] || boxVolume.first > grid->_max[i])
			return false; // No intersection possible
	}

	// Test the triangle normal
	double triangleOffset = face->_normal.Dot(face->_vertices[0]->_pos);
	auto triVolume = project(grid, face->_normal);
	if (triVolume.second < triangleOffset || triVolume.first > triangleOffset)
		return false; // No intersection possible

	// Test the nine edge cross-products
	Vec3<double> edges[3];
	edges[0] = face->_vertices[0]->_pos - face->_vertices[1]->_pos;
	edges[1] = face->_vertices[1]->_pos - face->_vertices[2]->_pos;
	edges[2] = face->_vertices[2]->_pos - face->_vertices[0]->_pos;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			// The box normals are the same as it's edge tangents
			auto axis = edges[i].Cross(boxNormals[j]);
			auto boxVolume = project(grid, axis);
			auto triVolume = project(face, axis);
			if (boxVolume.second < triVolume.first || boxVolume.first > triVolume.second)
				return false; // No intersection possible
		}
	}
	// No separating axis found
	return true;
}

pair<double, double> GridMesh::project(Grid *grid, Vec3<double> normal)
{
	double min = DBL_MAX;
	double max = -DBL_MAX;
	Vec3<double> vertices[8];
	vertices[0].Set(grid->_min.x(), grid->_min.y(), grid->_min.z());
	vertices[1].Set(grid->_max.x(), grid->_min.y(), grid->_min.z());
	vertices[2].Set(grid->_min.x(), grid->_max.y(), grid->_min.z());
	vertices[3].Set(grid->_max.x(), grid->_max.y(), grid->_min.z());
	vertices[4].Set(grid->_min.x(), grid->_min.y(), grid->_max.z());
	vertices[5].Set(grid->_max.x(), grid->_min.y(), grid->_max.z());
	vertices[6].Set(grid->_min.x(), grid->_max.y(), grid->_max.z());
	vertices[7].Set(grid->_max.x(), grid->_max.y(), grid->_max.z());
	for (int i = 0; i < 8; i++) {
		auto value = normal.Dot(vertices[i]);
		if (value < min) min = value;
		if (value > max) max = value;
	}
	return make_pair(min, max);
}

pair<double, double> GridMesh::project(Face *face, Vec3<double> normal)
{
	double min = DBL_MAX; 
	double max = -DBL_MAX;
	for (int i = 0; i < 3; i++) {
		auto value = normal.Dot(face->_vertices[i]->_pos);
		if (value < min) min = value;
		if (value > max) max = value;
	}
	return make_pair(min, max);
}

void GridMesh::extractSkeleton(void)
{
	for (auto j : _mesh->_skeleton->_Joint) {
		auto pos = j->_pos;
		int si = (int)fmax(0, fmin(_res - 1, _res*pos[0]));
		int sj = (int)fmax(0, fmin(_res - 1, _res*pos[1]));
		int sk = (int)fmax(0, fmin(_res - 1, _res*pos[2]));
		auto id = si * _res * _res + sj * _res + sk;
		_grids[id]->_skeletonType = SkeletonType::SKELETON_NODE;
		_grids[id]->_boneId = 0;
		//_grids[id]->_boneId.push_back(0);
	}
	
	for (auto b : _mesh->_skeleton->_bones)
	{
		int x0 = b->_start[0] * _res, y0 = b->_start[1] * _res, z0 = b->_start[2] * _res, x1 = b->_end[0] * _res, y1 = b->_end[1] * _res, z1 = b->_end[2] * _res;
		int dx = abs(x1 - x0);
		int dy = abs(y1 - y0);
		int dz = abs(z1 - z0);
		int stepX = x0 < x1 ? 1 : -1;
		int stepY = y0 < y1 ? 1 : -1;
		int stepZ = z0 < z1 ? 1 : -1;
		
		double hypotenuse = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
		double tMaxX = hypotenuse * 0.5 / dx;
		double tMaxY = hypotenuse * 0.5 / dy;
		double tMaxZ = hypotenuse * 0.5 / dz;

		double tDeltaX = hypotenuse / dx;
		double tDeltaY = hypotenuse / dy;
		double tDeltaZ = hypotenuse / dz;
		
		while (x0 != x1 || y0 != y1 || z0 != z1) {
			if (tMaxX < tMaxY) {
				if (tMaxX < tMaxZ) {
					x0 = x0 + stepX;
					tMaxX = tMaxX + tDeltaX;
				}
				else if (tMaxX > tMaxZ) {
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;
				}
				else {
					x0 = x0 + stepX;
					tMaxX = tMaxX + tDeltaX;
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;
				}
			}
			else if (tMaxX > tMaxY) {
				if (tMaxY < tMaxZ) {
					y0 = y0 + stepY;
					tMaxY = tMaxY + tDeltaY;
				}
				else if (tMaxY > tMaxZ) {
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;
				}
				else {
					y0 = y0 + stepY;
					tMaxY = tMaxY + tDeltaY;
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;

				}
			}
			else {
				if (tMaxY < tMaxZ) {
					y0 = y0 + stepY;
					tMaxY = tMaxY + tDeltaY;
					x0 = x0 + stepX;
					tMaxX = tMaxX + tDeltaX;
				}
				else if (tMaxY > tMaxZ) {
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;
				}
				else {
					x0 = x0 + stepX;
					tMaxX = tMaxX + tDeltaX;
					y0 = y0 + stepY;
					tMaxY = tMaxY + tDeltaY;
					z0 = z0 + stepZ;
					tMaxZ = tMaxZ + tDeltaZ;
				}
			}

			auto id = x0 * _res * _res + y0 * _res + z0;
			_grids[id]->_skeletonType = SkeletonType::SKELETON_NODE;
			_grids[id]->_boneId = b->_id;
			//_grids[id]->_boneId.push_back(b->_id);
		}
	}
}

void GridMesh::scanline(void)
{
	_hashTable->sort(_mesh);

	auto dx = 1.0 / (double)_res;
	// 바운더리 노드 찾기   
	for (int i = 0; i < _res; i++) {
		for (int j = 0; j < _res; j++) {
			for (int k = 0; k < _res; k++) {
				auto id = i * _res * _res + j * _res + k;
				int si = (int)fmax(0, fmin(_hashRes - 1, _hashRes*(i * dx)));
				int sj = (int)fmax(0, fmin(_hashRes - 1, _hashRes*(j * dx)));
				int sk = (int)fmax(0, fmin(_hashRes - 1, _hashRes*(k * dx)));
				auto neighbors = _hashTable->getNeighbors(si, sj, sk, 1, 1, 1);		
				for (auto tri : neighbors) {
					auto f = _mesh->_faces[tri];
					/*
					if (_grids[id]->_type != GridType::EMPTY) {
						break;
					}*/
					//_grids[id]->_type = GridType::BOUNDARAY_NODE;
					//_grids[id]->_faces.push_back(f);
					//for (auto v : f->_vertices) v->_flag = true;
					if (_grids[id]->_type == GridType::BOUNDARAY_NODE) {
						_grids[id]->_faces.push_back(f);
					}
					else if (intersect(f, _grids[id])) {
						_grids[id]->_type = GridType::BOUNDARAY_NODE;
						_grids[id]->_faces.push_back(f);
					}
				}
			}
		}
	}

	vector<int> range;
	for (int i = 0; i < _res; i++) {
		for (int j = 0; j < _res; j++) {
			range.clear();
			for (int k = 0; k < _res-1; k++) {
				auto id_curr = i * _res * _res + j * _res + k;
				auto type_curr = _grids[id_curr]->_type;
				if (type_curr == GridType::BOUNDARAY_NODE) {
					auto id_next = i * _res * _res + j * _res + (k + 1);
					auto type_next = _grids[id_next]->_type;
					if (type_next == GridType::EMPTY) {
						range.push_back(k);
					}
				}
			}
			int size = range.size();
			if (size > 0 && size % 2 == 0) {
				for (int k = 0; k < range.size(); k+=2) {
					for (int l = range[k] + 1; l < range[k + 1]; l++) {
						auto id = i * _res * _res + j * _res + l;
						if(_grids[id]->_type == GridType::EMPTY)
							_grids[id]->_type = GridType::INTERIOR_NODE;
					}
				}
			}
		}
	}

	for (auto g : _grids)
	{
		if (g->_type == GridType::EMPTY || g->_type == GridType::OUTERIOR_NODE)
		{
			int id = g->_i * _res * _res + g->_j * _res + g->_k;
			_emptyGridsId.push_back(id);
		}
	}
}

void GridMesh::init(void)
{
	auto dx = 1.0 / (double)_res;

	for (int i = 0; i < _res; i++) {
		for (int j = 0; j < _res; j++) {
			for (int k = 0; k < _res; k++) {
				auto p0 = _min + Vec3<double>(dx * i, dx * j, dx * k);
				auto p1 = _min + Vec3<double>(dx * (i + 1), dx * (j + 1), dx * (k + 1));
				auto g = new Grid(p0, p1);
				g->id(i, j, k);
				_grids.push_back(g);
			}
		}
	}
}

void GridMesh::gpuDijkstra(int source)
{
	int totalSize = _res * _res * _res;
	vector<int> voxelTypes(totalSize, NORMAL);
	vector<double> distances;
	vector<int> parents;

	// Place start point
	voxelTypes[source] = SKELETON;
	
	// Place AIR Node
	int count = 0;
	for (auto eid : _emptyGridsId)
	{
		voxelTypes[eid] = AIR;
		count++;
	}
	
	// Run CUDA Dijkstra
	runCUDADijkstra(voxelTypes, distances, parents, _res, _res, _res);

	// Copy Distances to Grid
	for (int i = 0; i < _res * _res * _res; i++)
	{
		_grids[i]->_distance = distances[i];
	}
	//cout << "distances[137315] : " << _grids[137315]->_distance << endl;
}
/*
void GridMesh::constructEdges(void)
{
	for (auto g : _grids)
	{
		if (g->_type == GridType::INTERIOR_NODE || g->_type == GridType::BOUNDARAY_NODE)
		{
			auto id0 = g->_i * _res * _res + g->_j * _res + g->_k;
			int nids[3];
			nids[0] = (g->_i + 1) * _res * _res + g->_j * _res + g->_k;
			nids[1] = g->_i * _res * _res + (g->_j + 1) * _res + g->_k;
			nids[2] = g->_i * _res * _res + g->_j * _res + (g->_k + 1);
			for (int n = 0; n < 3; n++)
			{
				if (_grids[nids[n]]->_type == GridType::INTERIOR_NODE || _grids[nids[n]]->_type == GridType::BOUNDARAY_NODE)
				{
					auto to = _grids[nids[n]];
					Vec3<double> diff = g->_center - to->_center;
					double distance = diff.Length();
					auto e = g->Connect(to, distance);
					_edges.push_back(e);
				}
			}
		}
	}
}*/

void GridMesh::calculateWeight(int boneId)
{
	float a = 0.7; // 애니메이터가 바인딩 부드러움을 제어할 수 있도록 하는 [0, 1] 범위의 매개변수
	double alpha = 0.2;
	double maxWeight = -1.0f;
	//double beta = 0.5;
	for (auto v : _mesh->_vertices)
	{
		v->_flag = false;
	}
	for (auto g : _grids)
	{
		if (g->_type == GridType::BOUNDARAY_NODE)
		{
			for (auto f : g->_faces)
			{
				for (auto v : f->_vertices)
				{
					if (!v->_flag)
					{
						Vec3<double> vvd = g->_center - v->_pos; // vertex to grid->center distance
						double d = (g->_distance + vvd.Length()) / _mesh->_boundDistance;
						double transformedD = (1 / d);
						//if (d < 0.07) d = 0.07;
						v->_weight[boneId] += pow((1 / ((1 - a) * d + a * d * d)), 2);
						//v->_weight = log(1/d);
						//v->_weight[boneId] += pow(1 / (1 + exp(-alpha * 1 / d)), 20);
						//v->_weight = pow(d, 3);
						//v->_weight[boneId] += pow(1 - d, 20);
						//v->_weight[boneId] += d;
						//v->_weight[boneId] += pow((1 - a)*(1 - d) + a * pow(1 - d, 2), 2);
						//v->_flag = true;
						//maxWeight = fmax(v->_weight[boneId], maxWeight);
					}
				}
			}
		}
	}
	//_maxWeight.push_back(maxWeight);
}

void GridMesh::exportWeight(int boneNum)
{
	FILE* fp = fopen("VertexWeight.csv", "w");
	for (auto v : _mesh->_vertices)
	{
		fprintf(fp, "%d", v->_index);
		for (int i = 0; i < boneNum; i++)
		{
			/*
			if (_maxWeight[i] != 0) v->_weight[i] = v->_weight[i];//v->_weight[i] /= _maxWeight[i];
			else v->_weight[i] = v->_weight[i - 1];*/
			
			fprintf(fp, ",%lf", v->_weight[i]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void GridMesh::findPath(void)
{
	cout << "Running CUDA Dijkstra..." << endl;
	//constructEdges();
	//Graph *graph = new Graph(_grids,/* _edges,*/ _res);
	//vector<int> sources;
	//int* d_sources = nullptr;
	
	for (auto g : _grids)
	{
		if (g->_skeletonType == SkeletonType::SKELETON_NODE)
		{
			int id = g->_i * _res * _res + g->_j * _res + g->_k;
			gpuDijkstra(id);
			//sources.push_back(id);
			//Path* route = graph->Find(id);
			calculateWeight(g->_boneId);
			/*
			for (auto boneId : g->_boneId)
			{
				calculateWeight(boneId);
			}*/
			// _grids[id]->_faces' _vertex 계산
			//delete route;
		}
	}
	/*
	int M = sources.size();
	int N = _grids.size();

	graph->copyToDeivice();
	cudaMalloc(&d_sources, sources.size() * sizeof(int));
	cudaMemcpy(d_sources, sources.data(), sources.size() * sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid(M);
	dim3 block(256);
	*/
	int boneNum = _mesh->_skeleton->_Joint.size();
	for (int i = 0; i < boneNum; i++)
	{
		double maxWeight = -1.0f;
		for (auto v : _mesh->_vertices)
		{
			maxWeight = fmax(v->_weight[i], maxWeight);
		}
		_maxWeight.push_back(maxWeight);
	}
	cout << "_maxWeight[0] : " << _maxWeight[0] << endl;
	exportWeight(boneNum);

	//int source = 1041982;
	//int source = 749884;
	/*
	int source = 1339708;
	Path* route = graph->Find(source);
	calculateWeight();*/
	/*
	int destination = _res * _res * _res;
	
	for (auto g : _grids)
	{
		if (g->_type == GridType::BOUNDARAY_NODE)
		{
			int id = g->_i * _res * _res + g->_j * _res + g->_k;
			if (abs(source - id) < abs(destination - id))
			{
				destination = id;
			}
		}
	}
	cout << "_grids[" << destination << "] : " << _grids[destination]->_distance << endl;
	bool traversble = route->Traverse(graph, destination, _resultGrids);*/
}
/*
void GridMesh::drawEdge(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_LINES);
	for (const auto& e : _edges) {
		auto p0 = e->_from->_center;
		auto p1 = e->_to->_center;
		glVertex3f(p0.x(), p0.y(), p0.z());
		glVertex3f(p1.x(), p1.y(), p1.z());
	}
	glEnd();
	glEnable(GL_LIGHTING);
	glPopMatrix();
}*/

void GridMesh::drawPath(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	for (int i = _resultGrids.size() - 2; i >= 0; i--)
	{
		auto p0 = _resultGrids[i + 1]->_center;
		auto p1 = _resultGrids[i]->_center;
		glVertex3f(p0.x(), p0.y(), p0.z());
		glVertex3f(p1.x(), p1.y(), p1.z());
	}
	glEnd();
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void GridMesh::draw(void)
{
	for (auto g : _grids) {
		glDisable(GL_LIGHTING);
		
		if (g->_type == GridType::INTERIOR_NODE) {
			glColor3f(0.0f, 1.0f, 0.0f);
			g->drawPoint();
		} else if (g->_type == GridType::BOUNDARAY_NODE) {
			glColor3f(0.0f, 0.5f, 1.0f);
			g->draw();
		}
		
		if (g->_skeletonType == SkeletonType::SKELETON_NODE /*&& g->_boneId == 0*/) {
			glColor3f(1.0f, 0.0f, 1.0f);
			g->draw();
		}/*
		if (g == _grids[1041982]) {
			glColor3f(0.0f, 0.0f, 1.0f);
			g->drawPoint();
		}
		else if (g == _grids[1978685]) {
			glColor3f(0.0f, 0.0f, 1.0f);
			g->drawPoint();
		}*/
		glEnable(GL_LIGHTING);
	}
}