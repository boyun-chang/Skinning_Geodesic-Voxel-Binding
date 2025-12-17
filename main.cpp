#include <stdio.h>
#include <Windows.h>
#include "GL\glut.h"
#include "Mesh.h"


int _width = 1024;
int _height = 800;
float _zoom = 1.099999;
float _rotate_x = 0.0f;
float _rotate_y = 0.001f;
float _translate_x = 0.0f;
float _translate_y = 0.0f;
int last_x = 0;
int last_y = 0;
bool _capture = false;
unsigned char _btnStates[3] = { 0 };
Mesh *_mesh;
bool _smoothing = false;

void init(void)
{
	glEnable(GL_DEPTH_TEST);
}

void Capture(void)
{
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	unsigned char *image = (unsigned char*)malloc(sizeof(unsigned char)*_width*_height * 3);
	FILE *file;
	string filename = "capture\\test.jpg";
	fopen_s(&file, filename.c_str(), "wb");
	if (image != NULL) {
		if (file != NULL) {
			glReadPixels(0, 0, _width, _height, 0x80E0, GL_UNSIGNED_BYTE, image);
			memset(&bf, 0, sizeof(bf));
			memset(&bi, 0, sizeof(bi));
			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf) + sizeof(bi) + _width * _height * 3;
			bf.bfOffBits = sizeof(bf) + sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = _width;
			bi.biHeight = _height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = _width * _height * 3;
			fwrite(&bf, sizeof(bf), 1, file);
			fwrite(&bi, sizeof(bi), 1, file);
			fwrite(image, sizeof(unsigned char), _height*_width * 3, file);
			fclose(file);
		}
		free(image);
	}
	_capture = false;
}

void draw(void)
{	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);	
	//_mesh->drawPoint();
	//_mesh->drawSurface(_smoothing);
	//_mesh->drawSkeleton();
	//_mesh->drawVoxel();
	_mesh->drawWeight();
	//_mesh->_gridMesh->drawEdge();
	//_mesh->_gridMesh->drawPath();
	glDisable(GL_LIGHTING);
}

void Domain(void)
{
	glPushMatrix();
	glLineWidth(3.0f);
	glColor3f(1.0f, 0.0f, 0.0f);
	//glTranslatef(0.5f, 0.5f, 0.5f);
	glutWireCube(1.0);
	glLineWidth(1.0f);
	glColor3f(1.0f, 1.0f, 1.0f);
	glPopMatrix();
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glLoadIdentity();

 	glTranslatef(0.0f, 0.0f, -_zoom); 
	glTranslatef(_translate_x, _translate_y, 0.0);
	glRotatef(_rotate_x, 1, 0, 0);
	glRotatef(_rotate_y, 0, 1, 0);
	glTranslatef(-0.5f, -0.5f, -0.5f);
	draw();
	if (_capture) {
		Capture();
	}
	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (float)w / (float)h, 0.1f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
}

void Motion(int x, int y)
{
	int diff_x = x - last_x;
	int diff_y = y - last_y;

	last_x = x;
	last_y = y;

	if (_btnStates[2]) {
		_zoom -= (float) 0.05f * diff_x;
	}
	else if (_btnStates[0]) {
		_rotate_x += (float)0.5f * diff_y;
		_rotate_y += (float)0.5f * diff_x;
	}
	else if (_btnStates[1]) {
		_translate_x += (float)0.05f * diff_x;
		_translate_y -= (float)0.05f * diff_y;
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	last_x = x;
	last_y = y;
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		_btnStates[0] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_MIDDLE_BUTTON:
		_btnStates[1] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_RIGHT_BUTTON:
		_btnStates[2] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'Q':
	case 'q':
		exit(0);
	case 'S':
	case 's':
		_smoothing = !_smoothing;
		break;
	case 'C':
	case 'c':
		_capture = true;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}


void main(int argc, char **argv)
{
	//_mesh = new Mesh("obj\\test.obj");
	//_mesh = new Mesh("obj\\TestModel.obj");
	_mesh = new Mesh("obj\\TestModel_4.obj");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(800, 800);
	glutInitWindowSize(_width, _height);
	glutCreateWindow("Geodesic Voxel Binding");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);
	init();
	glutMainLoop();
}