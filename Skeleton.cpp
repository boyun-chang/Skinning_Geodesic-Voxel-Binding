#include "Skeleton.h"

void Skeleton::loadTransform(const char *file)
{
	FILE *fp;
	char name[100] = { 0 };
	Vec3<double> pos;
	Hierarchy hierarchy;
	hierarchy.child_idx = -1;
	hierarchy.parent_idx = -1;

	fopen_s(&fp, file, "r");

	while (fscanf(fp, "%s (%lf, %lf, %lf)", name, &pos[0], &pos[1], &pos[2]) != EOF)
	{
		_Joint.push_back(new Joint(pos, name, hierarchy));
	}

	fclose(fp);
	moveToCenter(0.9);
}

void Skeleton::loadHierarchy(const char *file)
{
	FILE* fp;
	char name[100] = { 0 };
	int child_idx = -1, parent_idx = -1;

	fopen_s(&fp, file, "r");
	while (fscanf(fp, "%d %d %*s %s", &child_idx, &parent_idx, name) != EOF)
	{
		for (auto j : _Joint)
		{
			if (name == j->_rigname)
			{
				j->_hierarchy.child_idx = child_idx;
				j->_hierarchy.parent_idx = parent_idx;
				break;
			}
		}
	}

	fclose(fp);
	matchBone();
}

void Skeleton::moveToCenter(double scale)
{
	Vec3<double> origin(0.5, 0.5, 0.5);

	for (auto j : _Joint) {
		auto pos = j->_pos;
		auto centerToVertex = pos - _center;
		centerToVertex /= _longestLength;
		centerToVertex *= scale;
		pos = origin;
		pos += centerToVertex;
		j->_pos = pos;
	}
}

void Skeleton::matchBone(void)
{
	for (auto p : _Joint)
	{
		for (auto c : _Joint)
		{
			if (p->_hierarchy.child_idx == c->_hierarchy.parent_idx)
			{
				Bone *bone = new Bone;
				bone->_start = p->_pos;
				bone->_end = c->_pos;
				bone->_id = p->_hierarchy.child_idx;
				_bones.push_back(bone);
			}
		}
	}
}

void Skeleton::draw(void)
{
	drawPoint();
}

void Skeleton::drawPoint(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(5.0f);
	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_POINTS);
	for (auto j : _Joint) {
		auto pos = j->_pos;
		glVertex3f(pos.x(), pos.y(), pos.z());
	}
	glEnd();
	glEnable(GL_LIGHTING);
	glPointSize(1.0f);
	glPopMatrix();
}

void Skeleton::drawSkeleton(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glLineWidth(5.0f);
	glColor3f(1.0f, 1.0f, 0.0f);
	for (auto b : _bones)
	{
		glBegin(GL_LINES);
		glVertex3f(b->_start[0], b->_start[1], b->_start[2]);
		glVertex3f(b->_end[0], b->_end[1], b->_end[2]);
		glEnd();
	}
	glEnable(GL_LIGHTING);
	glLineWidth(1.0f);
	glPopMatrix();
}