#include "Face.h"


Face::Face()
{
}


Face::~Face()
{
}

void Face::center(void)
{
	_center = (_vertices[0]->_pos + _vertices[1]->_pos + _vertices[2]->_pos) / 3.0;
}

void Face::AABB(Vec3<double> &min, Vec3<double> &max)
{
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			min(j) = fmin(min(j), _vertices[i]->_pos(j));
			max(j) = fmax(max(j), _vertices[i]->_pos(j));
		}
	}
}