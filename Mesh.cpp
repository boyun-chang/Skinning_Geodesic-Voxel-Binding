#include "Mesh.h"
#include "GL\glut.h"
#include <chrono>

Mesh::Mesh()
{
}


Mesh::~Mesh()
{
}

void Mesh::loadObj(char *file)
{
	FILE *fp;
	int index[3], id;
	char buffer[100] = { 0 };
	Vec3<double> pos;
	Vec3<double> _minBound, _maxBound;
	_minBound.Set(DBL_MAX);
	_maxBound.Set(DBL_MIN);

	// read vertices 
	id = 0;
	fopen_s(&fp, file, "r");
	while (fscanf(fp, "%s %lf %lf %lf", &buffer, &pos[0], &pos[1], &pos[2]) != EOF) {
		if (buffer[0] == 'v' && buffer[1] == NULL) {
			if (_minBound[0] > pos[0]) _minBound[0] = pos[0];
			if (_minBound[1] > pos[1]) _minBound[1] = pos[1];
			if (_minBound[2] > pos[2]) _minBound[2] = pos[2];
			if (_maxBound[0] < pos[0]) _maxBound[0] = pos[0];
			if (_maxBound[1] < pos[1]) _maxBound[1] = pos[1];
			if (_maxBound[2] < pos[2]) _maxBound[2] = pos[2];
			_vertices.push_back(new Vertex(id++, pos));
		}
	}
	printf("num. of vertices : %d\n", _vertices.size());
	
	// read faces
	id = 0;
	fseek(fp, 0, SEEK_SET);
	while (fscanf(fp, /*"%s %d %d %d"*/"%s %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &buffer, &index[0], &index[1], &index[2]) != EOF) {
		if (buffer[0] == 'f' && buffer[1] == NULL) {
			_faces.push_back(new Face(id++, _vertices[index[0] - 1], _vertices[index[1] - 1], _vertices[index[2] - 1]));
		}
	}
	printf("num. of faces : %d\n", _faces.size());
	fclose(fp);

	moveToCenter(_minBound, _maxBound, 0.95f);
	buildList();
	computeNormal();
	
	_gridMesh = new GridMesh(this, 65);
	_gridMesh->extractSkeleton();
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	_gridMesh->findPath();
	std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
	printf("%f sec\n", sec.count());
}

void Mesh::buildList(void)
{
	for (auto f : _faces) {
		for (auto v : f->_vertices) {
			v->_nbFaces.push_back(f);
		}
	}
}

void Mesh::computeNormal(void)
{
	// face normal
	for (auto f : _faces) {
		auto v01 = f->_vertices[1]->_pos - f->_vertices[0]->_pos;
		auto v02 = f->_vertices[2]->_pos - f->_vertices[0]->_pos;
		f->_normal = v01.Cross(v02);
		f->_normal.Normalize();
	}

	// vertex normal
	for (auto v : _vertices) {
		v->_normal.Clear();
		for (auto f : v->_nbFaces) {
			v->_normal += f->_normal;
		}
		v->_normal /= (double)v->_nbFaces.size();
	}
}

void Mesh::drawSurface(bool smoothing)
{
	glPushMatrix();
	glEnable(GL_LIGHTING);	
	if (smoothing) {
		glEnable(GL_SMOOTH);
	} else {
		glEnable(GL_FLAT); // default
	}
	for (auto f : _faces) {
		glBegin(GL_POLYGON);
		if (smoothing) {
			for (auto v : f->_vertices) {
				glNormal3f(v->_normal.x(), v->_normal.y(), v->_normal.z());
				glVertex3f(v->x(), v->y(), v->z());
			}
		}
		else {
			//if (f->_flag) 
			{
				glNormal3f(f->_normal.x(), f->_normal.y(), f->_normal.z());
				for (auto v : f->_vertices) {
					glVertex3f(v->x(), v->y(), v->z());
				}
			}
		}
		glEnd();
	}	
	glPointSize(1.0f);
	glDisable(GL_LIGHTING);
	glDisable(GL_FLAT);
	glPopMatrix();
}

void Mesh::drawSkeleton(void)
{
	_skeleton->drawSkeleton();
	_skeleton->draw();
}

void Mesh::drawVoxel(void)
{
	_gridMesh->draw();
}

Vec3<float> Mesh::SCALAR_TO_COLOR(float scalar)
{
	scalar = std::clamp(scalar, 0.0f, 1.0f); // 값 제한 (0.0 ~ 1.0)

	// Hue를 0.66 (파랑)에서 0.0 (빨강)으로 매핑
	//float hue = (1.0f - scalar) * 0.66f; // 0.66은 파란색(Hue 기준)
	//float x = 1 - fabs(fmod(hue * 6.0f, 2.0f) - 1);

	float r, g, b;
	
	/*
	if (hue < 1.0f / 6.0f) {
		r = 1.0; g = x; b = 0;
	}
	else if (hue < 2.0f / 6.0f) {
		r = x; g = 1.0; b = 0;
	}
	else if (hue < 3.0f / 6.0f) {
		r = 0; g = 1.0; b = x;
	}
	else if (hue < 4.0f / 6.0f) {
		r = 0; g = x; b = 1.0;
	}
	else if (hue < 5.0f / 6.0f) {
		r = x; g = 0; b = 1.0;
	}
	else {
		r = 1.0; g = 0; b = x;
	}*/
	
	if (scalar < 0.2f) {
		// 검정 (0.0 ~ 0.2)
		r = 0.0f;
		g = 0.0f;
		b = 0.0f;
	}
	else if (scalar < 0.4f) {
		// 검정 → 파랑 (0.2 ~ 0.4)
		float t = (scalar - 0.2f) / 0.2f;
		r = 0.0f;
		g = 0.0f;
		b = 1.0f;
	}
	else if (scalar < 0.6f) {
		// 파랑 → 초록 (0.4 ~ 0.6)
		float t = (scalar - 0.4f) / 0.2f;
		r = 0.0f;
		g = t;
		b = 1.0f - t;
	}
	else if (scalar < 0.8f) {
		// 초록 → 노랑 (0.6 ~ 0.8)
		float t = (scalar - 0.6f) / 0.2f;
		r = t;
		g = 1.0f;
		b = 0.0f;
	}
	else {
		// 노랑 → 빨강 (0.8 ~ 1.0)
		float t = (scalar - 0.8f) / 0.2f;
		r = 1.0f;
		g = 1.0f - t;
		b = 0.0f;
	}
	
	return Vec3(r, g, b);
}
/*
void Mesh::clampWeight(void)
{
	double maxWeight = -1.0f;
	for (auto v : _vertices)
	{
		maxWeight = fmax(v->_weight[7], maxWeight);
	}
	
	for (auto v : _vertices)
	{
		v->_weight[7] /= maxWeight;
		//v->_weight /= 300;
	}
}*/
/*
template <typename T> inline Vec3<T> SCALAR_TO_COLOR(T val)
{
	// T fColorMap[3][3] = {{0.960784314,0.498039216,0.011764706},{0,0,0},{0,0.462745098,0.88627451}};
	T fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
	//T fColorMap[5][3] = { { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 1, 1 }, { 0, 0, 1 } };
	//T fColorMap[4][3] = {{0.15,0.35,1.0},{1.0,1.0,1.0},{1.0,0.5,0.0},{0.0,0.0,0.0}};
	T v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	T t = v - low;
	Vec3<T> color;
	color.x((fColorMap[low][0])*(1 - t) + (fColorMap[high][0])*t);
	color.y((fColorMap[low][1])*(1 - t) + (fColorMap[high][1])*t);
	color.z((fColorMap[low][2])*(1 - t) + (fColorMap[high][2])*t);
	return color;
}*/

void Mesh::drawWeight(void)
{
	glPushMatrix();
	glEnable(GL_LIGHTING);
	glEnable(GL_SMOOTH);
	float diffuse[4];
	//FILE* fp = fopen("VertexWeight.csv", "w");
	for (auto f : _faces) {
		glBegin(GL_POLYGON);
		for (auto v : f->_vertices) {
			auto color = SCALAR_TO_COLOR(v->_weight[64]);
			diffuse[0] = color.x();
			diffuse[1] = color.y();
			diffuse[2] = color.z();
			diffuse[3] = 1.0;
			//fprintf(fp, "%d %lf\n", v->_index, v->_weight);
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
			glNormal3f(v->_normal.x(), v->_normal.y(), v->_normal.z());
			glVertex3f(v->x(), v->y(), v->z());
		}
		glEnd();
	}/*
	for (auto v : _vertices)
	{
		fprintf(fp, "%d, %lf\n", v->_index, v->_weight[7]);
	}
	fclose(fp);*/
	glPointSize(1.0f);
	glDisable(GL_LIGHTING);
	glDisable(GL_FLAT);
	glPopMatrix();
}

void Mesh::drawPoint(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
	//glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
	//FILE* fp = fopen("VertexWeight.txt", "w");
	for (auto v : _vertices) {
		if (v->_flag) glColor3f(1.0f, 1.0f, 1.0f);
		else glColor3f(1.0f, 0.0f, 0.0f);
		auto pos = v->_pos;
		//glColor3f(v->_weight / 300, v->_weight / 300, v->_weight / 300);
		//fprintf(fp, "%d %lf\n", v->_index, v->_weight);
		//cout << "v->_weight : " << v->_weight << endl;
		glVertex3f(pos.x(), pos.y(), pos.z());
	}
	//fclose(fp);
	glEnd();
	glPointSize(1.0f);
	glDisable(GL_LIGHTING);
	glPopMatrix();
}

void Mesh::moveToCenter(Vec3<double> minBound, Vec3<double> maxBound, double scale)
{
	double longestLength = fmaxf(fmaxf(fabs(maxBound.x() - minBound.x()), fabs(maxBound.y() - minBound.y())), fabs(maxBound.z() - minBound.z()));
	auto center = (maxBound + minBound) / 2.0f;
	Vec3<double> origin(0.5, 0.5, 0.5);
	Vec3<double> bound = (maxBound - minBound) * scale / longestLength;
	_boundDistance = bound.Length();

	//_skeleton = new Skeleton("data\\skeleton_transforms.txt", "data\\Ch44_nonPBR_hierarchy.txt", longestLength, center);
	_skeleton = new Skeleton("data\\skeleton_transforms_4.txt", "data\\Ch19_nonPBR_hierarchy.txt", longestLength, center);

	for (auto v : _vertices) {
		auto pos = v->_pos;
		auto vecVertexFromCenter = pos - center;
		vecVertexFromCenter /= longestLength; //  <= 1
		vecVertexFromCenter *= scale; // <= 1*scale
		pos = origin;
		vecVertexFromCenter *= scale; // <= 1*scale
		pos += vecVertexFromCenter;
		v->_pos = pos;
		for (int i = 0; i <= _skeleton->_bones.size(); i++)
		{
			v->_weight.push_back(0);
		}
	}
}