#ifndef __MESH_H__
#define __MESH_H__

#pragma once
#include "Face.h"
#include "GridMesh.h"
#include "Skeleton.h"
#include <algorithm>

using namespace std;

class Mesh
{
public:
	vector<Face*>	_faces;
	vector<Vertex*>	_vertices;
	GridMesh		*_gridMesh;
	Skeleton		*_skeleton;
	double			_boundDistance;
public:
	Mesh();
	Mesh(char *file)
	{
		loadObj(file);
		//clampWeight();
	}
	~Mesh();
public:
	Vec3<float>	SCALAR_TO_COLOR(float scalar);
	//void	clampWeight(void);
public:
	void	loadObj(char *file);
	void	buildList(void);
	void	computeNormal(void);
	void	moveToCenter(Vec3<double> minBound, Vec3<double> maxBound, double scale);
public:
	void	drawVoxel(void);
	void	drawPoint(void);
	void	drawSkeleton(void);
	void	drawSurface(bool smoothing = false);
	void	drawWeight(void);
};

#endif