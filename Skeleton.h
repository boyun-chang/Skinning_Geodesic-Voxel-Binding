#ifndef __SKELETON_H__
#define __SKELETON_H__

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "Vec3.h"
#include "GL/glut.h"
#include "Joint.h"

using namespace std;

typedef struct
{
	Vec3<double>	_start;
	Vec3<double>	_end;
	int				_id;
} Bone;

class Skeleton
{
public:
	vector<Joint*>	_Joint;
	vector<Bone*>	_bones;
	double			_longestLength;
	Vec3<double>	_center;
public:
	Skeleton(void) {}
	Skeleton(const char *file_T, const char *file_H, double longestLength, Vec3<double>	center)
	{
		_longestLength = longestLength;
		_center = center;
		loadTransform(file_T);
		loadHierarchy(file_H);
	}
	~Skeleton(void) {}
public:
	void loadTransform(const char *file);
	void loadHierarchy(const char *file);
	void moveToCenter(double scale);
	void matchBone(void);
public:
	void draw(void);
	void drawPoint(void);
	void drawSkeleton(void);
};

#endif