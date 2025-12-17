#ifndef __VERTEX_H__
#define __VERTEX_H__

#pragma once
#include <vector>	
#include "Vec3.h"
using namespace std;

class Face;
class Vertex
{
public:
	int				_index;
	vector<double>	_weight;
	bool			_flag;
	Vec3<double>	_normal;
	Vec3<double>	_pos;
	vector<Face*>	_nbFaces; // neighboring face..
public:
	Vertex();
	Vertex(int index, Vec3<double> pos)
	{
		_index = index;
		_pos = pos;
		_flag = false;
	}
	~Vertex();
public:
	inline double x(void) { return _pos[0]; }
	inline double y(void) { return _pos[1]; }
	inline double z(void) { return _pos[2]; }
};

#endif