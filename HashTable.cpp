#include "HashTable.h"
#include "Mesh.h"

HashTable::HashTable(void)
{
}

HashTable::HashTable(int res)
{
	_res = res;
	_indices = Alloc3D<vector<int>>(res, res, res);
}

HashTable::~HashTable(void)
{
	Free3D(_indices);
}

void HashTable::clear(void)
{
	for (int i = 0; i < _res; i++) {
		for (int j = 0; j < _res; j++) {
			for (int k = 0; k < _res; k++) {
				_indices[i][j][k].clear();
			}
		}
	}
}

void HashTable::sort(Mesh *mesh)
{
	clear();

	// center
	for (auto f : mesh->_faces) {
		f->center();
		auto pos = f->_center;
		int i = (int)fmax(0, fmin(_res - 1, _res*pos[0]));
		int j = (int)fmax(0, fmin(_res - 1, _res*pos[1]));
		int k = (int)fmax(0, fmin(_res - 1, _res*pos[2]));
		_indices[i][j][k].push_back(f->_index);
	}
	printf("hash table : sort!\n");
}

vector<int>	HashTable::getForwardNeighbors(int i, int j, int k, int w, int h)
{
	vector<int> res;
	for (int si = i - w; si <= i + w; si++) {
		for (int sj = j - h; sj <= j + h; sj++) {
			for (int sk = 0; sk <= _res-1; sk++) {
				if (si < 0 || si > _res - 1 || sj < 0 || sj > _res - 1 || sk < 0 || sk > _res - 1) {
					continue;
				}
				for (int a = 0; a < (int)_indices[si][sj][sk].size(); a++) {
					int p = _indices[si][sj][sk][a];
					res.push_back(p);
				}
			}
		}
	}
	return res;
}

vector<int>	HashTable::getNeighbors(int i, int j, int k, int w, int h, int d)
{
	vector<int> res;
	for (int si = i - w; si <= i + w; si++) {
		for (int sj = j - h; sj <= j + h; sj++) {
			for (int sk = k - d; sk <= k + d; sk++) {
				if (si < 0 || si > _res - 1 || sj < 0 || sj > _res - 1 || sk < 0 || sk > _res - 1) {
					continue;
				}
				for (int a = 0; a < (int)_indices[si][sj][sk].size(); a++) {
					int p = _indices[si][sj][sk][a];
					res.push_back(p);
				}
			}
		}
	}
	return res;
}
