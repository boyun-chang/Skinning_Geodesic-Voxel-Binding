#ifndef __FACE_H__
#define __FACE_H__

#pragma once
#include "Vertex.h"

class Face
{
public:
	bool			_flag;
	int				_index;
	Vec3<double>	_normal;
	Vec3<double>	_center;
	vector<Vertex*>	_vertices;
public:
	Face();
	Face(int index, Vertex *v0, Vertex *v1, Vertex *v2)
	{
		_flag = false;
		_index = index;
		_vertices.push_back(v0);
		_vertices.push_back(v1);
		_vertices.push_back(v2);
	}
	~Face();
public:
	void			center(void);
	void			AABB(Vec3<double> &min, Vec3<double> &max);
};

#endif