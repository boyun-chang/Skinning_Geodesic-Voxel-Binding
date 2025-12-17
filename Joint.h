#ifndef __JOINT_H__
#define __JOINT_H__

#pragma once

#include <string>
#include "Vec3.h"

using namespace std;

typedef struct
{
	int parent_idx;
	int child_idx;
} Hierarchy;

class Joint
{
public:
	Vec3<double> _pos;
	string _rigname;
	Hierarchy _hierarchy;
public:
	Joint(void) {}
	Joint(Vec3<double> pos, string rigname, Hierarchy hierarchy)
	{
		_pos = pos;
		_rigname = rigname;
		_hierarchy.child_idx = hierarchy.child_idx;
		_hierarchy.parent_idx = hierarchy.parent_idx;
	}
	~Joint(void) {}
};

#endif