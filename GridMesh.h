#ifndef __GRID_MESH_H__
#define __GRID_MESH_H__

#pragma once
#include "Vec3.h"
#include <vector>
#include <utility>
#include "HashTable.h"
#include "voxel_dijkstra.h"

using namespace std;

enum class GridType {
	EMPTY, INTERIOR_NODE, OUTERIOR_NODE, BOUNDARAY_NODE
};

enum class SkeletonType {
	EMPTY, SKELETON_NODE
};


class Mesh;
class Face;
class Grid
{
public:	
	GridType		_type = GridType::EMPTY;
	SkeletonType	_skeletonType = SkeletonType::EMPTY;
	Vec3<double>	_min, _max, _center;
	vector<Face*>	_faces;
	Grid			*_prev = nullptr;
	double			_distance;
	int				_i, _j, _k;
	int				_boneId;
	
public:
	Grid(Vec3<double> min, Vec3<double> max) {
		_min = min;
		_max = max;
		_center = (_min + _max) / 2.0;
	}
public:
	void	id(int i, int j, int k);
	void	draw(void);
	void	drawPoint(void);
};

class GridMesh
{
public:
	int				_res;
	int				_hashRes;
	Vec3<double>	_min, _max;
	vector<double>	_maxWeight;
	vector<Grid*>	_grids;
	vector<Grid*>	_resultGrids;
	vector<int>		_emptyGridsId;
	Mesh			*_mesh;
	HashTable		*_hashTable;
public:
	GridMesh(Mesh *mesh, int res);
	~GridMesh(void);
public:				
	void					init(void);
	void					scanline(void);
	void					extractSkeleton(void);
	void					gpuDijkstra(int source);
	void					calculateWeight(int boneId);
	void					findPath(void);
	bool					intersect(Face *face, Grid *grid);
	pair<double, double>	project(Face *face, Vec3<double> normal);
	pair<double, double>	project(Grid *grid, Vec3<double> normal);
public:
	void					draw(void);
	void					drawEdge(void);
	void					drawPath(void);
	void					exportWeight(int boneNum);
};

#endif
