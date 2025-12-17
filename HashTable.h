#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

#pragma once
#include <cmath>
#include "Vec3.h"
#include <vector>

using namespace std;

template <typename T> T ***Alloc3D(int w, int h, int d)
{
	T *** Buffer = new T **[w + 1];
	for (int i = 0; i < w; i++) {
		Buffer[i] = new T*[h + 1];
		for (int j = 0; j < h; j++) {
			Buffer[i][j] = new T[d];
		}
		Buffer[i][h] = NULL;
	}
	Buffer[w] = NULL;
	return Buffer;
}

template <class T> void Free3D(T ***ptr)
{
	for (int i = 0; ptr[i] != NULL; i++) {
		for (int j = 0; ptr[i][j] != NULL; j++) delete[] ptr[i][j];
		delete[] ptr[i];
	}
	delete[] ptr;
}

class Mesh;
class HashTable
{
public:
	int			_res;
	vector<int>	***_indices;
public:
	HashTable(void);
	HashTable(int res);
	~HashTable(void);
public:
	void		clear(void);
	void		sort(Mesh *mesh);
	vector<int>	getNeighbors(int i, int j, int k, int w, int h, int d);
	vector<int>	getForwardNeighbors(int i, int j, int k, int w, int h);
};

#endif