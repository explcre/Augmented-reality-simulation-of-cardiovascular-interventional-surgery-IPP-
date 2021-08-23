// Lighthouse3D.com OpenGL 3.3 + GLSL 3.3 Sample
// http://www.lighthouse3d.com/cg-topics/code-samples/importing-3d-models-with-assimp/
// Lighthouse3D.com OpenGL 3.3 Loading an Image File and Creating a Texture
// http://www.lighthouse3d.com/2013/01/loading-and-image-file-and-creating-a-texture/
// C++ 11 Multithreading Tutorial
// https://solarianprogrammer.com/2011/12/16/cpp-11-thread-tutorial/
// AruCo Camera Calibration
// https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/calibrate_camera.cpp

//866 和1463

#ifdef _WIN32
#pragma comment(lib,"assimp.lib")
#pragma comment(lib,"devil.lib")
#pragma comment(lib,"glew32.lib")

#endif

//OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/opengl.hpp>



// include DevIL for image loading  包含DevIL进行图像加载
#include <IL/il.h>

// include GLEW to access OpenGL 3.3 functions  包含GLEW以访问OpenGL 3.3功能
#include <GL/glew.h>

// GLUT is the toolkit to interface with the OS  GLUT是与操作系统交互的工具包
#include <GL/freeglut.h>

// auxiliary C file to read the shader text files
//辅助C文件以读取着色器文本文件
#include "textfile.h"

// assimp include files. These three are usually needed.  assimp包含文件，通常需要这三个。
#include "assimp/Importer.hpp"    //OO version Header!
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include <math.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <thread>         // std::thread
#include <map>
#define WINDOW_NAME "AR demo"
using namespace std;
#include<GL/glut.h>
#include<windows.h>
#include<math.h>
#include<stdlib.h>
#include<iostream>
#define    C(j)   cos((3.1415926/180)*2*(j))
#define    S(j)   sin((3.1415926/180)*2*(j))
#define B 32
#pragma comment(lib, "glut32.lib") 
double ratio2 = 0.01;//血管的缩放比例
int upperbound = 1354;
GLfloat roate = 0.0;// 设置旋转速率
GLfloat rote = 0.0;//旋转角度

GLfloat anglex =-203.0;//X 轴旋转
GLfloat angley = 150.0;//Y 轴旋转
GLfloat anglez = -15.0;//Z 轴旋转

GLint WinW = 400;
GLint WinH = 400;
GLfloat oldx;//当左键按下时记录鼠标坐标  
GLfloat oldy;
int di = 10;//精确度，di越小越精确，di是每次跳过几个点
float PI = 3.1415926f;
double thick = 1.5;
int begini = 1;//renderscene里面画血管，循环开始的第i个点
double slice = 10.0;//血管的slices数
double stack = 4.0;//血管的stack数

double dx = 0.0, dy = 0.0, dz = 0.0;
//文件
FILE* fp = nullptr;
int timePin=1;//随时间变化，模型会改变


//Scene and view info for each model
//每个模型的场景和视图信息

struct MyModel
{
	aiScene* scene; // the global Assimp scene object for each model 每个模型的全局Assimp场景对象
	std::vector<cv::Mat> viewMatrix = { cv::Mat::zeros(4, 4, CV_32F),cv::Mat::zeros(4, 4, CV_32F),cv::Mat::zeros(4, 4, CV_32F) };
	int seenlast = 0;
	bool seennow = false;
	Assimp::Importer importer; // Create an instance of the Importer class创建Importer类的实例
	std::vector<struct MyMesh> myMeshes;
	float scaleFactor; // scale factor for the model to fit in the window  模型的比例因子以适合窗口
	int marker; //marker corresponding to this model  与此模型对应的标记
};

// Information to render each assimp node  渲染每个assimp节点的信息
struct MyMesh
{
	GLuint vao;
	GLuint texIndex;
	GLuint uniformBlockIndex;
	int numFaces;
};

// This is for a shader uniform block  这是用于着色器统一块
struct MyMaterial
{
	float diffuse[4];
	float ambient[4];
	float specular[4];
	float emissive[4];
	float shininess;
	int texCount;
};

class point3d {
public:
	float x;
	float y;
	float z;
	point3d() {}
	void assignValue(float x, float y, float z);
	~point3d() {}
};
//血管的点集
point3d* pt = new point3d[1400];



//Window Default size
int windowWidth = 512, windowHeight = 512;
//int tx_origin[2] = { 546,1000 }, ty_origin[2] = { 1146 ,1000 }, tz_origin[2] = { 1000,1000 };//坐标的初始值
int tx[2] = { 839,1000 }, ty[2] = { 1439 ,1000 }, tz[2] = { 1000,1000 };//改变量,第一个是血管模型，第二个是
int scalex[2] = { 1834,2000 }, scaley[2] = { 1093,2000 }, scalez[2] = { 2615,2000 };
int rotatexyz[2] = { 0,0 }, rotatex[2] = { 0,0 }, rotatey[2]= { 0,0 }, rotatez[2]= { 0,0 };
//GLdouble anglex=0.0, angley =0.0, anglez = 0.0;//血管摄像头的角度
int angle[2] = { 180, 90};
/*
int tx_dot = 1000, ty_dot = 1000, tz_dot = 1000;//改变量
int scalex_dot = 2000, scaley_dot = 2000, scalez_dot = 2000;
int rotatexyz_dot = 0, rotatex_dot, rotatey_dot, rotatez_dot;
int angle_dot = 90;*/
// Model Matrix (part of the OpenGL Model View Matrix)
float modelMatrix[16];

// For push and pop matrix
std::vector<float*> matrixStack;

// Vertex Attribute Locations 顶点属性位置
GLuint vertexLoc = 0, normalLoc = 1, texCoordLoc = 2;

// Uniform Bindings Points  统一绑定点
GLuint matricesUniLoc = 1, materialUniLoc = 2;

// The sampler uniform for textured models 用于纹理模型的采样器制服
// we are assuming a single texture so this will  我们假设一个纹理
//always be texture unit 0  始终为纹理单位0

GLuint texUnit = 0;

// Uniform Buffer for Matrices   矩阵的统一缓冲区
// this buffer will contain 3 matrices: projection, view and model        此缓冲区将包含3个矩阵：投影，视图和模型
// each matrix is a float array with 16 components      每个矩阵都是具有16个组件的float

GLuint matricesUniBuffer;
#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

// Program and Shader Identifiers    程序和着色器标识符
GLuint program, vertexShader, fragmentShader;
GLuint p, vertexShader2D, fragmentShader2D;


// holder for the vertex array object id    顶点数组对象ID的持有人
GLuint vao, textureID;


// Shader Names
char* vertexFileName = (char*)"dirlightdiffambpix.vert";
char* fragmentFileName = (char*)"dirlightdiffambpix.frag";

std::map<int, MyModel> models;


// images / texture
// map image filenames to textureIds
// pointer to texture Array
//图片/纹理
//将图像文件名映射到textureIds
//指向纹理数组的指针

std::map<std::string, GLuint> textureIdMap;

// Replace the model name by your model's filename
//static const std::string modelname = "jeep1.ms3d";

std::map<int, string> modelMap;
static const std::string modelDir = "F:/project/ar/models/";                  //3d文件路径


//our aruco variables
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

std::vector< int > markerIds;

std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;

//cv::VideoCapture cap("D:/profiles/心血管比赛相关资料/video_20201103_132337.mp4");
cv::VideoCapture cap("F:/project/test6.mp4");

bool flipped = false;

double K_[3][3] =
{ { 6.5423193332302969e+02, 0., 3.4382422833024395e+02 },
{ 0.,6.5358529192792446e+02, 2.7602128088484807e+02 },
{ 0, 0, 1 } };
cv::Mat K = cv::Mat(3, 3, CV_64F, K_).clone();
const float markerLength = 3.00;

// Distortion coeffs (fill in your actual values here). 失真系数（在此处填写您的实际值）。
double dist_[] = { 0, 0, 0, 0, 0 };
cv::Mat distCoeffs = cv::Mat(5, 1, CV_64F, dist_).clone();
cv::Mat imageMat;
cv::Mat imageMatGL;

#ifndef M_PI
#define M_PI       3.14159265358979323846f
#endif


static inline float
DegToRad(float degrees)
{
	return (float)(degrees * (M_PI / 180.0f));
};

// Frame counting and FPS computation  帧计数和FPS计算
long timet, timebase = 0, frame = 0;
char s[32];

//-----------------------------------------------------------------
// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.


#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char* file, int line)
{

	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	if (glErr != GL_NO_ERROR)
	{
		printf("glError in file %s @ line %d: %s\n",
			file, line, gluErrorString(glErr));
		retCode = 1;
	}
	return retCode;
}

// ----------------------------------------------------
// MATRIX STUFF
//

// Push and Pop for modelMatrix 

void pushMatrix()
{

	float* aux = (float*)malloc(sizeof(float) * 16);
	memcpy(aux, modelMatrix, sizeof(float) * 16);
	matrixStack.push_back(aux);
}

void popMatrix() {

	float* m = matrixStack[matrixStack.size() - 1];
	memcpy(modelMatrix, m, sizeof(float) * 16);
	matrixStack.pop_back();
	free(m);
}

// sets the square matrix mat to the identity matrix, 将平方矩阵垫设置为单位矩阵，
// size refers to the number of rows (or columns)   size是指行（或列）数

void setIdentityMatrix(float* mat, int size) {

	// fill matrix with 0s
	for (int i = 0; i < size * size; ++i)
		mat[i] = 0.0f;

	// fill diagonal with 1s  对角线
	for (int i = 0; i < size; ++i)
		mat[i + i * size] = 0.3f;                  //调整模型的大小 size
}


// a = a * b;

void multMatrix(float* a, float* b)
{

	float res[16];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			res[j * 4 + i] = 0.0f;
			for (int k = 0; k < 4; ++k) {
				res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
			}
		}
	}
	memcpy(a, res, 16 * sizeof(float));

}


// Defines a transformation matrix mat with a translation  定义带有转换的转换矩阵
void setTranslationMatrix(float* mat, float x, float y, float z)
{
	setIdentityMatrix(mat, 4);
	mat[12] = x;
	mat[13] = y;
	mat[14] = z;
}

// Defines a transformation matrix mat with a scale  用比例尺定义转换矩阵
void setScaleMatrix(float* mat, float sx, float sy, float sz)
{

	setIdentityMatrix(mat, 4);
	mat[0] = sx;
	mat[5] = sy;
	mat[10] = sz;
}

// Defines a transformation matrix mat with a rotation   定义旋转矩阵
// angle alpha and a rotation axis (x,y,z)  角度alpha和旋转轴（x，y，z）


void setRotationMatrix(float* mat, float angle, float x, float y, float z) {

	float radAngle = DegToRad(angle);
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x * x;
	float y2 = y * y;
	float z2 = z * z;

	mat[0] = x2 + (y2 + z2) * co;
	mat[4] = x * y * (1 - co) - z * si;
	mat[8] = x * z * (1 - co) + y * si;
	mat[12] = 0.0f;

	mat[1] = x * y * (1 - co) + z * si;
	mat[5] = y2 + (x2 + z2) * co;
	mat[9] = y * z * (1 - co) - x * si;
	mat[13] = 0.0f;

	mat[2] = x * z * (1 - co) - y * si;
	mat[6] = y * z * (1 - co) + x * si;
	mat[10] = z2 + (x2 + y2) * co;
	mat[14] = 0.0f;

	mat[3] = 0.0f;
	mat[7] = 0.0f;
	mat[11] = 0.0f;
	mat[15] = 1.0f;

}

// ----------------------------------------------------
// Model Matrix 
//
// Copies the modelMatrix to the uniform buffer  将modelMatrix复制到统一缓冲区


void setModelMatrix()
{

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ModelMatrixOffset, MatrixSize, modelMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

// The equivalent to glTranslate applied to the model matrix

void translate(float x, float y, float z)
{

	float aux[16];
	setTranslationMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glRotate applied to the model matrix

void rotate(float angle, float x, float y, float z) {

	float aux[16];

	setRotationMatrix(aux, angle, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glScale applied to the model matrix

void scale(float x, float y, float z) {

	float aux[16];

	setScaleMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// ----------------------------------------------------
// Projection Matrix 
// Computes the projection Matrix and stores it in the uniform buffer


void buildProjectionMatrix(float fov, float ratio, float nearp, float farp) {

	float projMatrix[16];

	float f = 1.0f / tan(fov * (M_PI / 360.0f));

	setIdentityMatrix(projMatrix, 4);

	projMatrix[0] = f / ratio;
	projMatrix[1 * 4 + 1] = f;
	projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
	projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
	projMatrix[2 * 4 + 3] = -1.0f;
	projMatrix[3 * 4 + 3] = 0.0f;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

// ----------------------------------------------------
// View Matrix
//
// Computes the viewMatrix and stores it in the uniform buffer
// note: it assumes the camera is not tilted, 
// i.e. a vertical up vector along the Y axis (remember gluLookAt?
//注意：假设相机没有倾斜，
//即沿Y轴的垂直向上矢量


void setCamera(cv::Mat viewMatrix) {

	//Set these to make the view matrix happy

	viewMatrix.at<float>(0, 3) = 0;
	viewMatrix.at<float>(1, 3) = 0;
	viewMatrix.at<float>(2, 3) = 0;
	viewMatrix.at<float>(3, 3) = 1;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, (float*)viewMatrix.data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}


// ----------------------------------------------------------------------------

#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

void get_bounding_box_for_node(const aiNode* nd,
	aiVector3D* min,
	aiVector3D* max,
	aiScene* scene)

{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];

			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		get_bounding_box_for_node(nd->mChildren[n], min, max, scene);
	}
}

void get_bounding_box(aiVector3D* min, aiVector3D* max, aiScene* scene)
{
	min->x = min->y = min->z = 1e10f;
	max->x = max->y = max->z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, min, max, scene);
}


bool loadModelFile(string filename) {
	filename = modelDir + filename;
	//open file stream to the file
	ifstream infile(filename);

	//if opened successfully, read in the data
	if (infile.is_open()) {
		int marker;
		string file;

		while (infile >> file >> marker)
		{
			//把文件中 的marker序号保存进入vector markerIds[]中
			//markerIds.push_back(marker);
			cout << "markerIds is :" << marker << endl;
			for (int i = 0; i < markerIds.size(); i++) {
				cout <<" out Id" << i << " is "<< markerIds[i]<<endl;
			}
			modelMap[marker] = file;
		}
	}
	else {
		//if here, file was not opened correctly, notify user
		printf("Error opening file,%s exiting", filename.c_str());
		exit(0);
	}
	return true;
}



bool Import3DFromFile(const std::string& pFile, aiScene*& scene, Assimp::Importer& importer, float& scaleFactor)
{
	std::string fileDir = modelDir + pFile;
	//check if file exists
	std::ifstream fin(fileDir.c_str());
	if (!fin.fail()) {
		fin.close();
	}
	else {
		printf("Couldn't open file: %s\n", fileDir.c_str());
		printf("%s\n", importer.GetErrorString());
		return false;
	}

	scene = const_cast<aiScene*>(importer.ReadFile(fileDir, aiProcessPreset_TargetRealtime_Quality));


	// If the import failed, report it
	if (!scene)
	{
		printf("%s\n", importer.GetErrorString());
		return false;
	}

	// Now we can access the file's contents.
	printf("Import of scene %s succeeded.\n", fileDir.c_str());

	aiVector3D scene_min, scene_max, scene_center;
	get_bounding_box(&scene_min, &scene_max, scene);
	float tmp;
	tmp = scene_max.x - scene_min.x;
	tmp = scene_max.y - scene_min.y > tmp ? scene_max.y - scene_min.y : tmp;
	tmp = scene_max.z - scene_min.z > tmp ? scene_max.z - scene_min.z : tmp;
	scaleFactor = 1.f / tmp;

	// We're done. Everything will be cleaned up by the importer destructor
	return true;
}

int LoadGLTextures(aiScene* scene)
{
	ILboolean success;

	/* initialization of DevIL */
	ilInit();

	/* scan scene's materials for textures 扫描场景的材质以获取纹理*/
	for (unsigned int m = 0; m < scene->mNumMaterials; ++m)
	{
		int texIndex = 0;
		aiString path;    // filename

		aiReturn texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		while (texFound == AI_SUCCESS) {
			//fill map with textures, OpenGL image ids set to 0
			textureIdMap[path.data] = 0;
			// more textures?
			texIndex++;
			texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		}
	}

	int numTextures = textureIdMap.size();

	/* create and fill array with DevIL texture ids 用DevIL纹理ID创建并填充数组*/
	ILuint* imageIds = new ILuint[numTextures];
	ilGenImages(numTextures, imageIds);

	/* create and fill array with GL texture ids  */
	GLuint* textureIds = new GLuint[numTextures];
	glGenTextures(numTextures, textureIds); /* Texture name generation */

											/* get iterator */
	std::map<std::string, GLuint>::iterator itr = textureIdMap.begin();
	int i = 0;
	for (; itr != textureIdMap.end(); ++i, ++itr)
	{
		//save IL image ID
		std::string filename = (*itr).first;  // get filename
		std::replace(filename.begin(), filename.end(), '\\', '/'); //Replace backslash with forward slash so linux can find the files
		filename = modelDir + filename;
		(*itr).second = textureIds[i];      // save texture id for filename in map

		ilBindImage(imageIds[i]); /* Binding of DevIL image name */
		ilEnable(IL_ORIGIN_SET);
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
		success = ilLoadImage((ILstring)filename.c_str());

		if (success) {
			/* Convert image to RGBA */
			ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);

			/* Create and load textures to OpenGL */
			glBindTexture(GL_TEXTURE_2D, textureIds[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ilGetInteger(IL_IMAGE_WIDTH),
				ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_RGBA, GL_UNSIGNED_BYTE,
				ilGetData());
		}
		else
			printf("Couldn't load Image: %s\n", filename.c_str());
	}
	/* Because we have already copied image data into texture data
	we can release memory used by image. */
	//因为我们已经将图像数据复制到纹理数据中
	//	我们可以释放图像使用的内存。 
	ilDeleteImages(numTextures, imageIds);

	//Cleanup
	delete[] imageIds;
	delete[] textureIds;

	//return success;
	return true;
}

//// Can't send color down as a pointer to aiColor4D because AI colors are ABGR.
//void Color4f(const aiColor4D *color)
//{
//    glColor4f(color->r, color->g, color->b, color->a);
//}

void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

void color4_to_float4(const aiColor4D* c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

void genVAOsAndUniformBuffer(aiScene* sc, std::vector<struct MyMesh>& myMeshes) {

	struct MyMesh aMesh;
	struct MyMaterial aMat;
	GLuint buffer;

	// For each mesh    对于每个啮合
	for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
	{
		const aiMesh* mesh = sc->mMeshes[n];

		// create array with faces
		// have to convert from Assimp format to array
		//创建带有面的数组
		//必须从Assimp格式转换为数组
		unsigned int* faceArray;
		faceArray = (unsigned int*)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
		unsigned int faceIndex = 0;

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];

			memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
			faceIndex += 3;
		}
		aMesh.numFaces = sc->mMeshes[n]->mNumFaces;

		// generate Vertex Array for mesh  生成网格的顶点数组
		glGenVertexArrays(1, &(aMesh.vao));
		glBindVertexArray(aMesh.vao);

		// buffer for faces  面部缓冲
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, GL_STATIC_DRAW);

		// buffer for vertex positions  顶点位置缓冲区
		if (mesh->HasPositions()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(vertexLoc);
			glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex normals 顶点法线的缓冲区
		if (mesh->HasNormals()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mNormals, GL_STATIC_DRAW);
			glEnableVertexAttribArray(normalLoc);
			glVertexAttribPointer(normalLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex texture coordinates  顶点纹理坐标的缓冲区
		if (mesh->HasTextureCoords(0)) {
			float* texCoords = (float*)malloc(sizeof(float) * 2 * mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

				texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
				texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;

			}
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * mesh->mNumVertices, texCoords, GL_STATIC_DRAW);
			glEnableVertexAttribArray(texCoordLoc);
			glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);
		}

		// unbind buffers 解除绑定缓冲区
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// create material uniform buffer  创建材料统一缓冲区
		aiMaterial* mtl = sc->mMaterials[mesh->mMaterialIndex];

		aiString texPath;    //contains filename of texture  
		if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)) {
			//bind texture
			unsigned int texId = textureIdMap[texPath.data];
			aMesh.texIndex = texId;
			aMat.texCount = 1;
		}
		else
			aMat.texCount = 0;

		float c[4];
		set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);
		memcpy(aMat.diffuse, c, sizeof(c));

		set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
		aiColor4D ambient;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
			color4_to_float4(&ambient, c);
		memcpy(aMat.ambient, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D specular;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
			color4_to_float4(&specular, c);
		memcpy(aMat.specular, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D emission;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
			color4_to_float4(&emission, c);
		memcpy(aMat.emissive, c, sizeof(c));

		float shininess = 0.0;
		unsigned int max;
		aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
		aMat.shininess = shininess;

		glGenBuffers(1, &(aMesh.uniformBlockIndex));
		glBindBuffer(GL_UNIFORM_BUFFER, aMesh.uniformBlockIndex);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(aMat), (void*)(&aMat), GL_STATIC_DRAW);

		myMeshes.push_back(aMesh);
	}
}


// ------------------------------------------------------------
//
// Reshape Callback Function重塑回调函数

//

void changeSize(int w, int h) {

	float ratio;
	// Prevent a divide by zero, when window is too short 当窗口太短时防止被零除
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	windowWidth = w;
	windowHeight = h;

	// Set the viewport to be the entire window  将视口设置为整个窗口
	glViewport(0, 0, w, h);
	ratio = w / h;
	buildProjectionMatrix(53.13f, ratio, 0.1f, 10.0f);

	//以下是血管新加的
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);//将当前矩阵设置为单位矩阵
	glLoadIdentity();//模型变换和视图变换
	gluPerspective(30.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);//类似眼睛，第一个参数为角度，决定能睁多大，第二个看到的最近距离第三个为最远距离
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);//前三个参数表示了观察点的位置，\
	//中间表示观察观察目标的位置，最后三个表示从（000）到（xyz）的直线，表示了观察者认为的"上"
}


// ------------------------------------------------------------
//
// Render stuff
//

// Render Assimp Model  渲染Assimp模型

void recursive_render(aiScene* sc, const aiNode* nd, std::vector<struct MyMesh>& myMeshes)
{

	// Get node transformation matrix  获取节点转换矩阵
	aiMatrix4x4 m = nd->mTransformation;
	// OpenGL matrices are column major
	m.Transpose();

	// save model matrix and apply node transformation   保存模型矩阵并应用节点变换
	pushMatrix();

	float aux[16];
	memcpy(aux, &m, sizeof(float) * 16);
	multMatrix(modelMatrix, aux);
	setModelMatrix();


	// draw all meshes assigned to this node  绘制分配给该节点的所有网格
	for (unsigned int n = 0; n < nd->mNumMeshes; ++n) {
		// bind material uniform  bind:绑定
		glBindBufferRange(GL_UNIFORM_BUFFER, materialUniLoc, myMeshes[nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MyMaterial));
		// bind texture
		glBindTexture(GL_TEXTURE_2D, myMeshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		glBindVertexArray(myMeshes[nd->mMeshes[n]].vao);
		// draw
		glDrawElements(GL_TRIANGLES, myMeshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);
		point3d* pt = {};
		//pt= assignBatchPoint();
		int ptsSize = 1354;

	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n) {
		recursive_render(sc, nd->mChildren[n], myMeshes);
	}
	popMatrix();
}


// Rendering Callback Function  渲染回调功能
//!!!!!!!
void detectArucoMarkers(cv::Mat& image)
{//调试输出
	cout << "begin detectArucoMarkers" << endl;
	for (int i = 0; i < markerIds.size(); i++) {
		cout << " out Id" << i << " is " << markerIds[i] << endl;
	}//调试输出
	

	
	unsigned int pinID = 1;//这个是传感器探针的markerid,这个探针会移动
	cv::aruco::detectMarkers(
		image,        // input image
		dictionary,        // type of markers that will be searched for
		markerCorners,    // output vector of marker corners
		markerIds,        // detected marker IDs
		detectorParams,    // algorithm parameters
		rejectedCandidates);
	map<int, MyModel>::iterator it;

	for (it = models.begin(); it != models.end(); it++) {
		it->second.seennow = false;
	}
	/*
	//以下三行是加的
	markerIds.push_back(0);//初始化markerIds
	markerIds.push_back(1);
	markerIds.push_back(2);*/

	if (markerIds.size() > 0) {
		// Draw all detected markers.

		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

		std::vector< cv::Vec3d > rvecs, tvecs;

		cv::aruco::estimatePoseSingleMarkers(
			markerCorners,    // vector of already detected markers corners
			markerLength,    // length of the marker's side
			K,                // input 3x3 floating-point instrinsic camera matrix K
			distCoeffs,        // vector of distortion coefficients of 4, 5, 8 or 12 elements
			rvecs,            // array of output rotation vectors 
			tvecs);            // array of output translation vectors

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Mat viewMatrix = cv::Mat::zeros(4, 4, CV_32F);;
			cv::Mat viewMatrixavg = cv::Mat::zeros(4, 4, CV_32F);
			cv::Vec3d r = rvecs[i];
			
			tvecs[i][0] = tvecs[i][0] + (double)tx[0] / 100 - (double)10 /*+ (double)pt[timePin].x/50.0*/;//pt[timePin].x是随时间变化的坐标
			tvecs[i][1] = tvecs[i][1] + (double)ty[0] / 100 - (double)20 /*+ (double)pt[timePin].y/50.0*/;
			tvecs[i][2] = tvecs[i][2] + (double)tz[0] / 100 - (double)10 /*+ (double)pt[timePin].z/50.0*/;
			if (markerIds[i] == pinID) {//这段程序是新加的，用于使探针的传感器位移按照pt[]集合 中的点变化
				tvecs[pinID][0]-= (double)pt[timePin].y/7.0 -60;//pt[timePin].x是随时间变化的坐标
				tvecs[pinID][1]-= -(double)pt[timePin].x/9.5;
				tvecs[pinID][2]-= (double)pt[timePin].z/5.0 -50;
			}
			cv::Vec3d t = tvecs[i];
			cv::Mat rot;
			Rodrigues(rvecs[i], rot);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{


					viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				viewMatrix.at<float>(row, 3) = (float)tvecs[i][row] * 0.1f;
			}
			viewMatrix.at<float>(3, 3) = 1.0f;

			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
			cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
			cvToGl.at<float>(3, 3) = 1.0f;
			viewMatrix = cvToGl * viewMatrix;
			cv::transpose(viewMatrix, viewMatrix);
			bool flipped = false;


			if (modelMap.count(markerIds[i])) {
				models[markerIds[i]].seennow = true;

				for (int jj = 0; jj < 4; jj++) {
					for (int yy = 0; yy < 4; yy++) {
						if (models[markerIds[i]].seenlast > 3 || models[markerIds[i]].seenlast < 0) {
							if (std::abs(viewMatrix.at<float>(jj, yy) - models[markerIds[i]].viewMatrix[1].at<float>(jj, yy)) > .5)flipped = true;
						}

						viewMatrixavg.at<float>(jj, yy) = (viewMatrix.at<float>(jj, yy) + models[markerIds[i]].viewMatrix[1].at<float>(jj, yy) + models[markerIds[i]].viewMatrix[2].at<float>(jj, yy)) / 3;
					}
				}
				if (models[markerIds[i]].seenlast < 0) {
					models[markerIds[i]].seenlast = 0;
				}

				if (flipped) {
					models[markerIds[i]].viewMatrix[0] = models[markerIds[i]].viewMatrix[1];
				}
				else {
					models[markerIds[i]].viewMatrix[0] = viewMatrixavg;
					models[markerIds[i]].viewMatrix[2] = models[markerIds[i]].viewMatrix[1];
					models[markerIds[i]].viewMatrix[1] = viewMatrix;
				}
			}

			// Draw coordinate axes.
			cv::aruco::drawAxis(image,
				K, distCoeffs,            // camera parameters
				r, t,                    // marker pose
				0.5 * markerLength);        // length of the axes to be drawn

										  // Draw a symbol in the upper right corner of the detected marker.
		}
	}

	for (it = models.begin(); it != models.end(); it++) {
		if (!it->second.seennow) {
			if (it->second.seenlast == -10) {
				it->second.viewMatrix[0] = cv::Mat::zeros(4, 4, CV_32F);
				it->second.seenlast = 0;
			}
			else if (it->second.seenlast < 0) {
				it->second.seenlast = it->second.seenlast - 1;
			}
			else if (it->second.seenlast > 0) {
				it->second.seenlast = -1;
			}

		}
		else if (it->second.seenlast < 100) {
			it->second.seenlast++;
		}
	}
}



void prepareTexture(int w, int h, unsigned char* data)
{

	/* Create and load texture to OpenGL */
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
		w, h,
		0, GL_RGB, GL_UNSIGNED_BYTE,
		data);
	glGenerateMipmap(GL_TEXTURE_2D);
}

void camTimer()
{

	cap >> imageMat; // get image from camera
	if (!imageMat.empty()) {
		// Convert to RGB
		cv::cvtColor(imageMat, imageMat, CV_BGR2RGB);

		if (flipped) {
			cv::flip(imageMat, imageMat, 0);
			cv::flip(imageMat, imageMat, 1);
		}
		detectArucoMarkers(imageMat);

		imageMat.copyTo(imageMatGL);

		camTimer();
	}

}



//两点之间画圆柱体
void RenderBone(float x0, float y0, float z0, float x1, float y1, float z1, GLdouble radius, GLdouble slices, GLdouble stack)
{
	GLdouble  dir_x = x1 - x0;
	GLdouble  dir_y = y1 - y0;
	GLdouble  dir_z = z1 - z0;
	GLdouble  bone_length = sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
	static GLUquadricObj* quad_obj = NULL;
	if (quad_obj == NULL)
		quad_obj = gluNewQuadric();
	gluQuadricDrawStyle(quad_obj, GLU_FILL);
	gluQuadricNormals(quad_obj, GLU_SMOOTH);
	glPushMatrix();
	// 平移到起始点
	glTranslated(x0, y0, z0);
	// 计算长度
	double  length;
	length = sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
	if (length < 0.0001) {
		dir_x = 0.0; dir_y = 0.0; dir_z = 1.0;  length = 1.0;
	}
	dir_x /= length;  dir_y /= length;  dir_z /= length;
	GLdouble  up_x, up_y, up_z;
	up_x = 0.0;
	up_y = 1.0;
	up_z = 0.0;
	double  side_x, side_y, side_z;
	side_x = up_y * dir_z - up_z * dir_y;
	side_y = up_z * dir_x - up_x * dir_z;
	side_z = up_x * dir_y - up_y * dir_x;
	length = sqrt(side_x * side_x + side_y * side_y + side_z * side_z);
	if (length < 0.0001) {
		side_x = 1.0; side_y = 0.0; side_z = 0.0;  length = 1.0;
	}
	side_x /= length;  side_y /= length;  side_z /= length;
	up_x = dir_y * side_z - dir_z * side_y;
	up_y = dir_z * side_x - dir_x * side_z;
	up_z = dir_x * side_y - dir_y * side_x;
	// 计算变换矩阵
	GLdouble  m[16] = { side_x, side_y, side_z, 0.0,
		up_x,   up_y,   up_z,   0.0,
		dir_x,  dir_y,  dir_z,  0.0,
		0.0,    0.0,    0.0,    1.0 };
	glMultMatrixd(m);
	// 圆柱体参数
	/*GLdouble radius = radius;		// 半径
	GLdouble slices = slices;		//	段数
	GLdouble stack = stack;		// 递归次数*/
	gluCylinder(quad_obj, radius, radius, bone_length, slices, stack);
	glPopMatrix();
}




void renderScene(void)
{//这里是更新时间变量
	if (timePin < 1353)
		timePin++;
	else
		timePin = 1;

	int ModelNo = 0;//这里0或1代表血管模型或者探针

	// Create Texture
	prepareTexture(imageMatGL.cols, imageMatGL.rows, imageMatGL.data);
	//gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, imageMat.cols, imageMat.rows, GL_RGB, GL_UNSIGNED_BYTE, imageMat.data);
	// clear the framebuffer (color and depth)  清除帧缓冲区（颜色和深度）
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	

	//DRAW 2D VIDEO
	// Use the program p
	glUseProgram(p);
	// Bind the vertex array object
	glBindVertexArray(vao);
	// Bind texture
	glBindTexture(GL_TEXTURE_2D, textureID);
	// draw the 6 vertices
	glDrawArrays(GL_TRIANGLES, 0, 6);


	
	//以下是画模型
	//DRAW 3D MODEL
	glClear(GL_DEPTH_BUFFER_BIT);
	// set camera matrix

	map<int, MyModel>::iterator it;
	for (it = models.begin(); it != models.end(); it++)
	{
		MyModel currentModel = it->second;

		setCamera(currentModel.viewMatrix[0]);

		// set the model matrix to the identity Matrix 将模型矩阵设置为恒等矩阵
		setIdentityMatrix(modelMatrix, 4);

		// sets the model matrix to a scale matrix so that the model fits in the window
		//模型矩阵设置为比例矩阵，以便模型适合窗口
		scale((double)scalex[ModelNo] / 2000 * currentModel.scaleFactor, (double)scaley[ModelNo] / 2000 * currentModel.scaleFactor, (double)scalez[ModelNo] / 2000 * currentModel.scaleFactor);
		rotatex[ModelNo] = rotatey[ModelNo] = rotatez[ModelNo] = 0;
		switch (rotatexyz[ModelNo])
		{
			case 0:rotatex[ModelNo] = 1; break;
			case 1:rotatey[ModelNo] = 1; break;
			case 2:rotatez[ModelNo] = 1; break;
		}
		// keep rotating the model  继续旋转模型
		rotate( (double)angle[ModelNo] * 1.0f , rotatex[ModelNo] * 1.0f, rotatey[ModelNo] * 1.0f, rotatez[ModelNo] * 1.0f);
		//原来是angle[0]
		// use our shadershader  使用我们的shadershader
		glUseProgram(program);

		// we are only going to use texture unit 0
		// unfortunately samplers can't reside in uniform blocks
		// so we have set this uniform separately
		//我们将仅使用纹理单元0
	   //不幸的是，采样器不能驻留在统一的块中
		//因此我们分别设置了此制服
		glUniform1i(texUnit, 0);

		//glLoadMatrixf((float*)viewMatrix.data);
		//注释了下面一行
		
	
		recursive_render(currentModel.scene, currentModel.scene->mRootNode, currentModel.myMeshes);

	}
	/*
	//以下是血管的程序
	//glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();  //加载单位矩阵  
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glRotatef(rote, 0.0f, 1.0f, 0.0f);//旋转矩阵
	glRotatef(anglex, 1.0, 0.0, 0.0);//旋转矩阵
	glRotatef(angley, 0.0, 1.0, 0.0);//旋转矩阵
	glRotatef(anglez, 0.0, 0.0, 1.0);//旋转矩阵
	glColor3f(1.0, 0.0, 0.0); //画笔红色
	//glColor3f(0.1, 0.1, 0.8); //画笔

	//画血管
	for (int i = begini; i < upperbound - di; i += di) {
		std::cout << "#" << i << "-th :    " << pt[i].x << "\n" << pt[i].y << "\n" << pt[i].z << "\n" << pt[i + 1].x << "\n" << pt[i + 1].y << "\n" << pt[i + 1].z << "\n";
		RenderBone((pt[i].x - dx) * ratio2, (pt[i].y - dy) * ratio2, (pt[i].z - dz) * ratio2, (pt[i + di].x - dx) * ratio2, (pt[i + di].y - dy) * ratio2, (pt[i + di].z - dz) * ratio2, thick * ratio2, slice, stack);
	}
	rote += roate;//血管增加旋转角度
	rote = rote - ((int)rote / 360) * 360;//若超过360度，取 除以360的余数
	*/


	/*
	// FPS computation and display  FPS计算和显示
	frame++;
	timet = glutGet(GLUT_ELAPSED_TIME);
	if (timet - timebase > 1000) {
		sprintf(s, "FPS:%4.2f , x=%f, y=%f, z=%f, anglex=%f, angley=%f, anglez=%f, scaleRatio=%f，roate= %f ,rote= %f, slices=%f,stacks=%f, thick = % f",
			frame * 1000.0 / (timet - timebase),dx,dy,dz, anglex, angley, anglez,ratio2,roate,rote,slice,stack,thick);
		timebase = timet;
		frame = 0;
		glutSetWindowTitle(s);
	}
	*/
	// swap buffers  交换缓冲区
	glutSwapBuffers();//交换缓冲区，使绘制的图形得以展示

}


// ------------------------------------------------------------
//
// Events from the Keyboard   键盘事件
//

void processKeys(unsigned char key, int xx, int yy)
{

	
	switch (key) {

	case 27:

		glutLeaveMainLoop();
		break;
	case 'm': glEnable(GL_MULTISAMPLE); break;
	case 'n': glDisable(GL_MULTISAMPLE); break;
	case 'f': flipped = !flipped; break;
	case '0': cap = cv::VideoCapture(0); break;
	case '1': cap = cv::VideoCapture(1); break;
	case '-': if (ratio2 > 0)ratio2 *= 0.8; break;
	case '+': ratio2 *= 1.2;   scaley[0] *= 1.2; break;
	case 'o': if (upperbound > 0) { upperbound -= (int)upperbound * 0.2; if (upperbound < 0) { upperbound = 0; }}  break;
	case 'p': if (upperbound < 1354){upperbound += (int)  upperbound * 0.2; 1.2; if (upperbound >= 1354) { upperbound = 1354; }} break;
	case 's': slice++; break;
	case 'd': if (slice > 1.0) { slice--; } break;
	case 't': stack++; break;
	case 'y': if (stack > 1.0) { stack--; } break;
	case 'i': if (di > 1354) { di++; } break;
	case 'k': if (di > 1) { di--; } break;
	case 'h': thick += 0.05; break;
	case 'g': if (thick > 0.05) { thick -= 0.05; } break;
	case 'q': exit(0);

	}
	glutPostRedisplay();
}


// --------------------------------------------------------
//
// Shader Stuff   材料着色器
//

void printShaderInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char* infoLog;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char*)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}


void printProgramInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char* infoLog;

	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char*)malloc(infologLength);
		glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}


GLuint setupShaders() {

	char* vs = NULL, * fs = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(vertexFileName);
	fs = textFileRead(fragmentFileName);

	const char* vv = vs;
	const char* ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindFragDataLocation(p, 0, "output");

	glBindAttribLocation(p, vertexLoc, "position");
	glBindAttribLocation(p, normalLoc, "normal");
	glBindAttribLocation(p, texCoordLoc, "texCoord");

	glLinkProgram(p);
	glValidateProgram(p);

	program = p;
	vertexShader = v;
	fragmentShader = f;

	GLuint k = glGetUniformBlockIndex(p, "Matrices");
	glUniformBlockBinding(p, k, matricesUniLoc);
	glUniformBlockBinding(p, glGetUniformBlockIndex(p, "Material"), materialUniLoc);

	texUnit = glGetUniformLocation(p, "texUnit");

	return(p);
}

// --------------------------------------------------------
//
//            Shader Stuff
//
// --------------------------------------------------------

void setupShaders2D() {

	// variables to hold the shader's source code   用于保存着色器源代码的变量
	char* vs = NULL, * fs = NULL;

	// holders for the shader's ids    着色器ID的持有人
	GLuint v, f;

	// create the two shaders   
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	// read the source code from file   
	vs = textFileRead((char*)"texture.vert");
	fs = textFileRead((char*)"texture.frag");

	// castings for calling the shader source function  用于调用着色器源函数的转换
	const char* vv = vs;
	const char* ff = fs;

	// setting the source for each shader
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	// free the source strings
	free(vs); free(fs);

	// compile the sources
	glCompileShader(v);
	glCompileShader(f);

	// create a program and attach the shaders
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	// Bind the fragment data output variable location
	// requires linking afterwards
	glBindFragDataLocation(p, 0, "outputF");

	// link the program
	glLinkProgram(p);

	GLint myLoc = glGetUniformLocation(p, "texUnit");
	glProgramUniform1d(p, myLoc, 0);
}



int init2D() {

	// Data for the two triangles
	float position[] = {
		1.0f, -1.0f, 0.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 1.0f,
		1.0f,  1.0f, 0.0f, 1.0f,

		-1.0f, 1.0f, 0.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f,
		-1.0f,  -1.0f, 0.0f, 1.0f,

	};

	float textureCoord[] = {
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,

		0.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

	};


	// variables to hold the shader's source code
	char* vs = NULL, * fs = NULL;

	// holders for the shader's ids
	GLuint v, f;

	// create the two shaders
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	// read the source code from file
	vs = textFileRead((char*)"texture.vert");
	fs = textFileRead((char*)"texture.frag");

	// castings for calling the shader source function
	const char* vv = vs;
	const char* ff = fs;

	// setting the source for each shader
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	// free the source strings
	free(vs); free(fs);

	// compile the sources
	glCompileShader(v);
	glCompileShader(f);

	// create a program and attach the shaders
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	// Bind the fragment data output variable location
	// requires linking afterwards
	glBindFragDataLocation(p, 0, "outputF");

	// link the program
	glLinkProgram(p);

	GLint myLoc = glGetUniformLocation(p, "texUnit");
	glProgramUniform1d(p, myLoc, 0);

	GLuint vertexLoc, texCoordLoc;

	// Get the locations of the attributes in the current program
	vertexLoc = glGetAttribLocation(p, "position");
	texCoordLoc = glGetAttribLocation(p, "texCoord");

	// Generate and bind a Vertex Array Object
	// this encapsulates the buffers used for drawing the triangle
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Generate two slots for the position and color buffers
	GLuint buffers[2];
	glGenBuffers(2, buffers);

	// bind buffer for vertices and copy data into buffer
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
	glEnableVertexAttribArray(vertexLoc);
	glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, 0, 0, 0);

	// bind buffer for normals and copy data into buffer
	glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoord), textureCoord, GL_STATIC_DRAW);
	glEnableVertexAttribArray(texCoordLoc);
	glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);


	glGenTextures(1, &textureID); //Gen a new texture and store the handle in texname

								  //These settings stick with the texture that's bound. You only need to set them
								  //once.
	glBindTexture(GL_TEXTURE_2D, textureID);

	//allocate memory on the graphics card for the texture. It's fine if
	//texture_data doesn't have any data in it, the texture will just appear black
	//until you update it.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, {});


	return true;
}



int loadModels() {


	map<int, string>::iterator it;

	for (it = modelMap.begin(); it != modelMap.end(); it++)
	{
		int markerNum = it->first;
		string modelName = it->second;

		models[markerNum].marker = markerNum;

		if (!Import3DFromFile(modelName, models[markerNum].scene, models[markerNum].importer, models[markerNum].scaleFactor))
			return(-1);

		LoadGLTextures(models[markerNum].scene);

		genVAOsAndUniformBuffer(models[markerNum].scene, models[markerNum].myMeshes);

	}


	return 0;
}


// ------------------------------------------------------------
//
// Model loading and OpenGL setup  模型加载和OpenGL设置
//


int init()
{
	glEnable(GL_BLEND); // 打开混合
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glEnable(GL_LIGHT1);
	glColor4f(1, 1, 1, 0.5);
	loadModels();
	//glDisable(GL_BLEND);


	glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)glutGetProcAddress("glGetUniformBlockIndex");
	glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)glutGetProcAddress("glUniformBlockBinding");
	glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glutGetProcAddress("glGenVertexArrays");
	glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glutGetProcAddress("glBindVertexArray");
	glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)glutGetProcAddress("glBindBufferRange");
	glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glutGetProcAddress("glDeleteVertexArrays");




	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 0.0f);

	//
	// Uniform Block
	//
	glGenBuffers(1, &matricesUniBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferData(GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, matricesUniLoc, matricesUniBuffer, 0, MatricesUniBufferSize);    //setUniforms();
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glEnable(GL_MULTISAMPLE);

	init2D();


	return true;
}


void myTimer(int value) {
	glutPostRedisplay();
	glutTimerFunc(1000.0f / 60.0f, myTimer, 0);

}
void initsliderwindow() {
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
	cv::createTrackbar("tx：", WINDOW_NAME, &tx[0], 2000, NULL);//滑动条创建
	cv::createTrackbar("ty：", WINDOW_NAME, &ty[0], 5000, NULL);
	cv::createTrackbar("tz：", WINDOW_NAME, &tz[0], 4000, NULL);
	cv::createTrackbar("size of x：", WINDOW_NAME, &scalex[0], 4000, NULL);//滑动条创建
	cv::createTrackbar("size of y：", WINDOW_NAME, &scaley[0], 4000, NULL);
	cv::createTrackbar("size of z：", WINDOW_NAME, &scalez[0], 4000, NULL);
	cv::createTrackbar("Direction of rotation：", WINDOW_NAME, &rotatexyz[0], 2, NULL);//滑动条创建
	//cv::createTrackbar("rotate of y：", WINDOW_NAME, &rotatey, 1,NULL);
	//cv::createTrackbar("rotate of z：", WINDOW_NAME, &rotatez, 1, NULL);
	cv::createTrackbar("angle：", WINDOW_NAME, &angle[0], 360, NULL);//



	//cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
	cv::createTrackbar("tx_dot：", WINDOW_NAME, &tx[1], 2000, NULL);//滑动条创建
	cv::createTrackbar("ty_dot：", WINDOW_NAME, &ty[1], 5000, NULL);
	cv::createTrackbar("tz_dot：", WINDOW_NAME, &tz[1], 4000, NULL);
	cv::createTrackbar("size of x_dot：", WINDOW_NAME, &scalex[1], 4000, NULL);//滑动条创建
	cv::createTrackbar("size of y_dot：", WINDOW_NAME, &scaley[1], 4000, NULL);
	cv::createTrackbar("size of z_dot：", WINDOW_NAME, &scalez[1], 4000, NULL);
	cv::createTrackbar("Direction of rotation_dot：", WINDOW_NAME, &rotatexyz[1], 2, NULL);//滑动条创建
	//cv::createTrackbar("rotate of y：", WINDOW_NAME, &rotatey, 1,NULL);
	//cv::createTrackbar("rotate of z：", WINDOW_NAME, &rotatez, 1, NULL);
	cv::createTrackbar("angle_dot：", WINDOW_NAME, &angle[1], 180, NULL);
}
// ------------------------------------------------------------
//
// Main function 
//

#include <stdio.h>  
#include <winsock2.h>  

#pragma comment(lib,"ws2_32.lib")  

int main_socket(int argc, char* argv[])
{
	//初始化WSA  
	WORD sockVersion = MAKEWORD(2, 2);
	WSADATA wsaData;
	if (WSAStartup(sockVersion, &wsaData) != 0)
	{
		return 0;
	}

	//创建套接字  
	SOCKET slisten = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (slisten == INVALID_SOCKET)
	{
		printf("socket error !");
		return 0;
	}

	//绑定IP和端口  
	sockaddr_in sin;
	sin.sin_family = AF_INET;
	sin.sin_port = htons(8888);
	sin.sin_addr.S_un.S_addr = INADDR_ANY;
	if (::bind(slisten, (LPSOCKADDR)&sin, sizeof(sin)) == SOCKET_ERROR)
	{
		printf("bind error !");
	}

	//开始监听  
	if (listen(slisten, 5) == SOCKET_ERROR)
	{
		printf("listen error !");
		return 0;
	}

	//循环接收数据  
	SOCKET sClient;
	sockaddr_in remoteAddr;
	int nAddrlen = sizeof(remoteAddr);
	char revData[255];
	while (true)
	{
		printf("等待连接...\n");
		sClient = accept(slisten, (SOCKADDR*)&remoteAddr, &nAddrlen);
		if (sClient == INVALID_SOCKET)
		{
			printf("accept error !");
			continue;
		}
		printf("接受到一个连接：%s \r\n",inet_ntoa(remoteAddr.sin_addr));

		//接收数据  
		int ret = recv(sClient, revData, 255, 0);
		if (ret > 0)
		{
			revData[ret] = 0x00;
			printf(revData);
		}

		//发送数据  
		const char* sendData = "你好，TCP客户端！\n";
		send(sClient, sendData, strlen(sendData), 0);
		closesocket(sClient);
	}

	closesocket(slisten);
	WSACleanup();
	return 0;
}



void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)//左键
	{
		if (state == GLUT_DOWN)//按下
		{
			
			//roate = 0;
			//rote = 0;

			oldx = x;//当左键按下时记录鼠标坐标  
			oldy = y;

			//ratio2 /= 2;
		}
	}
	if (button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			//angle[0] += 10.0;
			//scalex[0] *= 1.2;

			
			roate += 0.5f;
			
			//ratio2 *= 2;
		}
	}
}

void motion(int x, int y)
{
	GLint deltax = oldx - x;
	GLint deltay = oldy - y;
	anglex += 360 * (GLfloat)deltax / (GLfloat)WinW;//根据屏幕上鼠标滑动的距离来设置旋转的角度  
	angley += 360 * (GLfloat)deltay / (GLfloat)WinH;
	anglez += 360 * (GLfloat)deltay / (GLfloat)WinH;
	oldx = x;//记录此时的鼠标坐标，更新鼠标坐标  
	oldy = y;//若是没有这两句语句，滑动时旋转会变得不可控  
	glutPostRedisplay();//标记当前窗口需要重新绘制，通过glutMainLoop下一次循环，窗口将被回调重新显示
	glutPostRedisplay();
}



void myDisplay()//单独显示血管线
{/*
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0, 1.0, 0.0);
	glutSolidTeapot(200);

	GLUquadric* pObj;
	pObj = gluNewQuadric();

	mySolidCylinder(pObj,0.01,0.02,0.03,360,100);

	//gluDeleteQuadric(pObj);
	glFlush();*/


	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0); //画笔红色
	glLoadIdentity();  //加载单位矩阵  
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glRotatef(rote, 0.0f, 1.0f, 0.0f);//
	glRotatef(anglex, 1.0, 0.0, 0.0);
	glRotatef(angley, 0.0, 1.0, 0.0);
	glRotatef(anglez, 0.0, 0.0, 0.0);
	//
	//glutSolidTeapot(200);

	//GLUquadric* pObj;
	//pObj = gluNewQuadric();


	point3d* pt = new point3d[1400];
	
	FILE* fp = nullptr;
	if ((fp = fopen("F:\\Project\\ar\\centerline.txt", "r")) == NULL) //文件路径需改为下面模拟数据所导入的文件路径
	{
		printf("can't open this file");
		exit(0);
	}
	int count = 1;
	for (int count = 1; count < 1354; count++) {
		fscanf(fp, "%f%f%f", &pt[count].x, &pt[count].y, &pt[count].z);
	}
	double k = 1.0;


	int di = 2;

	double dx = 0.0, dy = 0.0, dz = 0.0;
	for (int i = 1; i < upperbound; i += di) {
		dx += pt[i].x;
		dy += pt[i].y;
		dz += pt[i].z;
	}dx = (double)dx / upperbound;
	dy = (double)dy / upperbound;
	dz = (double)dz / upperbound;

	dx = pt[1].x;
	dy = pt[1].y;
	dz = pt[1].z;
	
	for (int i = 1; i < upperbound; i += di) {
		std::cout << "#" << i << "-th :    " << pt[i].x << "\n" << pt[i].y << "\n" << pt[i].z << "\n" << pt[i + 1].x << "\n" << pt[i + 1].y << "\n" << pt[i + 1].z << "\n";
		RenderBone((pt[i].x - dx) * ratio2, (pt[i].y - dy) * ratio2, (pt[i].z - dz) * ratio2, (pt[i + di].x - dx) * ratio2, (pt[i + di].y - dy) * ratio2, (pt[i + di].z - dz) * ratio2, 0.3 * ratio2, 4.0, 1.0);
	}


	glColor3f(0.0, 0.0, 1.0); //画笔红色



	rote += roate;



	glutSwapBuffers();//交换缓冲区，使绘制的图形得以展示
	delete[] pt;
	fclose(fp);
	fp = NULL;
}

int main_model(int argc, char** argv) {

	//main2(argc,  argv);


	initsliderwindow();
	//  GLUT initialization
	//init();
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);

	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);


	glutInitWindowPosition(100, 100);
	windowWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	windowHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("Lighthouse3D - Assimp Demo");


	//  Callback Registration   
	//glutDisplayFunc(renderScene);
	
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);


	glutTimerFunc(1000.0f / 60.0f, myTimer, 0);
	std::thread t1(camTimer);
	//(1000.0f / 15.0f, camTimer, 0);

	//cap=cv::VideoCapture("D:/profiles/心血管比赛相关资料/video_20201103_132337.mp4");
	cap = cv::VideoCapture("F:/project/video_20201103_132337.mp4");
	//    Mouse and Keyboard Callbacks   
	glutMouseFunc(mouse);//用于注册键盘和鼠标事件发生时的回调函数
	//glutKeyboardFunc(keyFunc); //血管部分的键盘函数
	glutKeyboardFunc(processKeys);

		
	glutMotionFunc(motion);//注册鼠标移动事件的回调函数，人机交互处理
	glutIdleFunc(&renderScene);//在键盘按下时作某事

	//    Init GLEW
	//glewExperimental = GL_TRUE;
	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 not supported\n");
		return(1);
	}

	loadModelFile((string)"modelToMarker.txt");


	//  Init the app (load model and textures) and OpenGL
	if (!init())
		printf("Could not Load the Model\n");

	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));


	// return from main loop
	//glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	program = setupShaders();
	//setupShaders2D();

	//  GLUT main loop
	glutMainLoop();

	// cleaning up
	textureIdMap.clear();

	// clear myMeshes stuff  清除myMeshes内容
	map<int, MyModel>::iterator it;
	for (it = models.begin(); it != models.end(); it++)
	{
		MyModel currentModel = it->second;

		for (unsigned int j = 0; j < currentModel.myMeshes.size(); ++j) {
			glDeleteVertexArrays(1, &(currentModel.myMeshes[j].vao));
			glDeleteTextures(1, &(currentModel.myMeshes[j].texIndex));
			glDeleteBuffers(1, &(currentModel.myMeshes[j].uniformBlockIndex));
		}
	}
	// delete buffers
	glDeleteBuffers(1, &matricesUniBuffer);

	exit(0);
};

/**

int winWidth = 400, winHeight = 300;
//定义变量
int flag = 0;
int n = 0;
int tempX, tempY;


struct LineNode {
	point3d point1;
	point3d point2;
	int x1;
	int y1;
	int z1;
	int x2;
	int y2;
	int z2
}Line[1500];


void Initial(void) {
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void ChangeSize(int w, int h) {
	winWidth = w;
	winHeight = h;
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, winWidth, 0.0, winHeight);
}

void Display(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0f, 0.0f, 0.0f);

	int i = 0;
	//绘制折线
	for (i = 0; i < n; i++) {
		glBegin(GL_LINES);
		glVertex3i(Line[i].x1, Line[i].y1,Line[i].z1);
		glVertex3i(Line[i].x2, Line[i].y2, Line[i].z1);
		glEnd();

	}
	if (flag == 1) {
		glBegin(GL_LINES);
		glVertex2i(Line[i].x1, Line[i].y1);
		//glVertex2i(tempX, tempY);
		glVertex2i(Line[i].x2, Line[i].y2);
		glEnd();
	}

	glutSwapBuffers();
}


//响应鼠标函数
void MousePlot(GLint button, GLint action, GLint xMouse, GLint yMouse) {
	//左键点击
	if (button == GLUT_LEFT_BUTTON && action == GLUT_DOWN) {
		if (flag == 0) {
			flag = 1;
			Line[n].x1 = xMouse;
			Line[n].y1 = winHeight - yMouse;
			//cout<<Line[n].x1<<" "<<Line[n].y1<<endl;

		}
		else {
			//flag = 0;
			Line[n].x2 = xMouse;
			Line[n].y2 = winHeight - yMouse;

			n++;
			//
			Line[n].x1 = Line[n - 1].x2;
			Line[n].y1 = Line[n - 1].y2;
		}

	}
	//右键点击
	if (button == GLUT_RIGHT_BUTTON && action == GLUT_DOWN) {
		flag = 0;
		n = 0;
		glutPostRedisplay();

	}

}

// 鼠标移动函数
void PassiveMouseMove(GLint xMouse, GLint yMouse) {
	//
	//if(flag == 1){
		//tempX = xMouse;
		//tempY = winHeight - yMouse;
	Line[n].x2 = xMouse;
	Line[n].y2 = winHeight - yMouse;
	glutPostRedisplay();
	//Display();
	//}

}

//
int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(400, 300);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("drawline");
	glutDisplayFunc(connectPoints);

	glutReshapeFunc(ChangeSize);//回调函数
	glutMouseFunc(MousePlot);//指定鼠标点击响应函数
	glutPassiveMotionFunc(PassiveMouseMove);//指定鼠标移动响应函数
	Initial();
	glutMainLoop();
	return 0;
}**/








/**

#include <GL/glut.h>
#include <cstdio>
#include <cmath>

const GLfloat Pi = 3.1415926536f;

//定义点集
struct data {
	GLfloat x;
	GLfloat y;
	GLfloat z;
}Point[1500];

void init2()  //初始化函数
{
	glClearColor(1.0, 1.0, 1.0, 0.0); //设置背景颜色
	//glMatrixMode(GL_PROJECTION);       // 设置投影参数
	//gluOrtho2D(0.0, 50.0, 0.0, 50.0); // 设置场景的大小

}

void mydisplay()
{

	int ptsSize = 1354;
	point3d* pt = assignBatchPoint();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < ptsSize; i++)
	{
		glVertex3f(pt[i].x, pt[i].y, pt[i].z);
	}
	glEnd();
	glFlush();

	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0); //设置线条颜色
	glPointSize(2); //设置点的大小

	//glTranslatef(100.0f, 100.0f, 0.0f); //平移图形
	glScalef(0.005f, 0.005f, 0.005f); //缩小图像0.5倍
	//glRotatef(60.0f, 1.0f, 0.0f, 0.0f); //沿x轴旋转60度

	glBegin(GL_LINE_STRIP);
	for (int i = 1; i <= 1354; i++)
	{
		GLfloat t = i / 200.0;
		pt[i].x = 50.0 * cos(2.0 * Pi * t); //参数曲线
		pt[i].y = 50.0 * sin(2.0 * Pi * t); //参数曲线
		pt[i].z = 100.0 * t;            //参数曲线

		glVertex3i(pt[i].x, pt[i].y, pt[i].z); //绘制曲线
	}
	glEnd();


	glFlush();
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(400, 400);
	glutCreateWindow("绘制空间曲线");
	init2();
	glutDisplayFunc(&mydisplay);
	glutMainLoop();
	return 0;
}
**/





void point3d::assignValue( GLfloat x, GLfloat y, GLfloat z) {
	this->x = x;
	this->y = y;
	this->z = z;
}



//point3d* newpt = new point3d[1355];

//血管中心点的坐标数据
point3d* assignBatchPoint(point3d* pt) {
	//point3d pt[1355];

	FILE* fp = NULL;
	
	
	if ((fp = fopen("F:\\Project\\ar\\centerline.txt", "r")) == NULL) //文件路径需改为下面模拟数据所导入的文件路径
	{
		printf("can't open this file");
		exit(0);
	}

	for (int i = 1; i < 1354; i++) {
		fscanf(fp, "%f%f%f", &pt[i].x, &pt[i].y, &pt[i].z);
	}
	
	return pt;
}
/*
void init(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0); //背景黑色  
}*/

typedef struct cmpcolor
{
	float r;
	float g;
	float b;
}COL;
COL tucolor(float value)
{
	COL tubecolor = { 0 };
	value > (B + 0.1) ? (tubecolor.r = 0.6, tubecolor.g = 0.5 + (value - B) / 20.0, tubecolor.b = 0) : \
		(((value < B + 0.1 && (value > B - 0.1) ? (tubecolor.r = 1, tubecolor.g = 0.5, tubecolor.b = 0) : (tubecolor.r = 0.55, tubecolor.g = 0.7 - (B - value) / 20.0, tubecolor.b = 0.5))));
	return tubecolor;
}
float lagrange_polynomial(float* data, float radian_assum)
{
	float y = 0;
	int i = 0;
	int j = 0;
	float radian[7] = { 0,30,60,90,120,150,180 };
	float lagx[7] = { 1,1,1,1,1,1,1 };
	for (i = 0; i < 7; i++)
	{
		for (j = 0; j < 7; j++)
		{
			if (i == j)
			{
				continue;
			}
			lagx[i] = lagx[i] * (radian_assum - radian[j]) / (radian[i] - radian[j]);//拉格朗日基本多项式
		}
	}
	for (i = 0; i < 7; i++)
	{
		y = y + data[i] * lagx[i];//拉格朗日插值多项式
	}
	return y;
}
void lagrange_interpolation(float* input_data, float* radius)
{
	float data[7] = { 0 };
	int i = 0;
	for (i = 0; i < 6; i++)
	{
		data[i] = input_data[i];
	}
	data[6] = data[0];
	for (i = 0; i < 181; i++)
	{
		radius[i] = lagrange_polynomial(data, i);
	}
}
void caculate_seat()
{
	int j = 0;
	int i = 1;
	int k = 0;
	FILE* fp = NULL;
	float input_data[6] = { 0 };
	COL seatcolor = { 0 };
	float seat_1[181] = { 0 };
	if ((fp = fopen("F:\\Project\\ar\\allOne.txt", "r")) == NULL) //文件路径需改为下面模拟数据所导入的文件路径
	{
		printf("can't open this file");
		exit(0);
	}
	fscanf(fp, "%f%f%f%f%f%f", &input_data[0], &input_data[1], &input_data[2], &input_data[3], &input_data[4], &input_data[5]);
	//读一次数据给input数组
	lagrange_interpolation(input_data, seat_1);
	seat_1[180] = seat_1[0];
	float seat_2[181] = { 0 };
	for (i = 1; i < 200; i++)
	{
		fscanf(fp, "%f%f%f%f%f%f", &input_data[0], &input_data[1], &input_data[2], &input_data[3], &input_data[4], &input_data[5]);
		//读一次数据给input数组，
		lagrange_interpolation(input_data, seat_2);
		seat_2[180] = seat_2[0];
		for (j = 0; j < 180; j++)
		{
			glBegin(GL_TRIANGLES);
			seatcolor = tucolor(seat_1[j]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_1[j] * C(j) / B, seat_1[j] * S(j) / B, (-200 + 2 * (i - 1.0)) / B);
			seatcolor = tucolor(seat_2[j]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_2[j] * C(j) / B, seat_2[j] * S(j) / B, (-200 + 2.0 * i) / B);
			seatcolor = tucolor(seat_1[j + 1]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_1[j + 1] * C(j + 1) / B, seat_1[j + 1] * S(j + 1) / B, (-200 + 2.0 * (i - 1)) / B);//第一个三角
			glEnd();
			glBegin(GL_TRIANGLES);
			seatcolor = tucolor(seat_1[j + 1]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_1[j + 1] * C(j + 1) / B, seat_1[j + 1] * S(j + 1) / B, (-200 + 2.0 * (i - 1)) / B);
			seatcolor = tucolor(seat_2[j + 1]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_2[j + 1] * C(j + 1) / B, seat_2[j + 1] * S(j + 1) / B, (-200 + 2.0 * i) / B);
			seatcolor = tucolor(seat_2[j]);
			glColor3f(seatcolor.r, seatcolor.g, seatcolor.b);
			glVertex3f(seat_2[j] * C(j) / B, seat_2[j] * S(j) / B, (-200 + 2.0 * i) / B);	//第二个三角
			glEnd();
		}
		for (k = 0; k < 181; k++)
		{
			seat_1[k] = seat_2[k];
		}
		//		Sleep(20);
	}
	fclose(fp);
	fp = NULL;
}


GLvoid DrawCircleArea(float cx, float cy, float cz, float r, int num_segments)
{
	GLfloat vertex[4];

	const GLfloat delta_angle = 2.0 * PI / num_segments;
	glBegin(GL_TRIANGLE_FAN);

	vertex[0] = cx;
	vertex[1] = cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);

	//draw the vertex on the contour of the circle
	for (int i = 0; i < num_segments; i++)
	{
		vertex[0] = std::cos(delta_angle * i) * r + cx;
		vertex[1] = std::sin(delta_angle * i) * r + cy;
		vertex[2] = cz;
		vertex[3] = 1.0;
		glVertex4fv(vertex);
	}

	vertex[0] = 1.0 * r + cx;
	vertex[1] = 0.0 * r + cy;
	vertex[2] = cz;
	vertex[3] = 1.0;
	glVertex4fv(vertex);
	glEnd();
}
void mySolidCylinder(GLUquadric* quad,
	GLdouble base,
	GLdouble top,
	GLdouble height,
	GLint slices,
	GLint stacks)
{
	glColor3f(84.0 / 255, 0.0, 125.0 / 255.0);
	gluCylinder(quad, base, top, height, slices, stacks);
	//top
	DrawCircleArea(0.0, 0.0, height, top, slices);
	//base
	DrawCircleArea(0.0, 0.0, 0.0, base, slices);
}
void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);//将当前矩阵设置为单位矩阵
	glLoadIdentity();//模型变换和视图变换
	gluPerspective(30.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);//类似眼睛，第一个参数为角度，决定能睁多大，第二个看到的最近距离第三个为最远距离
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);//前三个参数表示了观察点的位置，\
	//中间表示观察观察目标的位置，最后三个表示从（000）到（xyz）的直线，表示了观察者认为的"上"
}
void keyFunc(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '-': if (ratio2 > 0)ratio2 *= 0.8; break;
	case '+': ratio2 *=1.2; break;
	case 'p': if(upperbound<1354)upperbound*=1.2; break;
	case 'q': exit(0);

	}
	glutPostRedisplay();
}
void init3(void)
{
	glClearColor(0.0, 0.0, 0.0, 1.0); //背景黑色  
}
int main_/*_vessel*/(int argc, char* argv[]){
	initsliderwindow();
	glutInit(&argc, argv);
	

	if ((fp = fopen("F:\\Project\\ar\\centerline.txt", "r")) == NULL) //文件路径需改为下面模拟数据所导入的文件路径
	{
		printf("can't open this file");
		exit(0);
	}
	//导入pt
	for (int count = 1; count < 1354; count++) {
		fscanf(fp, "%f%f%f", &pt[count].x, &pt[count].y, &pt[count].z);
	}
	//double k = 1.0;
	//给dx,dy,dz赋值

	for (int i = 1; i < upperbound; i += di) {
		dx += pt[i].x;
		dy += pt[i].y;
		dz += pt[i].z;
	}dx = (double)dx / upperbound;
	dy = (double)dy / upperbound;
	dz = (double)dz / upperbound;

	dx = pt[630].x;
	dy = pt[630].y;
	dz = pt[630].z;

	//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);//vessel
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);//model
	//glutInitContextVersion(3, 3);//model
	//glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);//model
	glutInitWindowPosition(200, 100);
	//glutInitWindowSize(2400, 2400);//vessel
	windowWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);//model
	windowHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	glutInitWindowSize(windowWidth, windowHeight);//model
	glutCreateWindow("Lighthouse3D - Assimp Demo+vessel diaplay");//model+vessel
	init3();
	glutDisplayFunc(&renderScene);
	glutReshapeFunc(reshape);//血管是reshape,模型是changeSize//用于注册窗口大小改变这一事件发生时GLUT将调用的函数
	glutMouseFunc(mouse);//用于注册键盘和鼠标事件发生时的回调函数

	glutKeyboardFunc(processKeys);
	glutMotionFunc(motion);//注册鼠标移动事件的回调函数，人机交互处理
	glutIdleFunc(&renderScene);//在键盘按下时作某事
	glutMainLoop();//事件处理函数，直至程序结束，点x关闭为结束标志
	delete[] pt;
	fclose(fp);
	fp = NULL;
	return 0;
}



//////////////////////以下是原来的main

int main(int argc, char** argv){


	initsliderwindow();
	//  GLUT initialization
	//init();
	glutInit(&argc, argv);
	
	/************************************************************************************************/
	//导入文件
	if ((fp = fopen("F:\\Project\\ar\\centerline.txt", "r")) == NULL) //文件路径需改为下面模拟数据所导入的文件路径
	{
		printf("can't open this file");
		exit(0);
	}
	//导入pt
	for (int count = 1; count < 1354; count++) {
		fscanf(fp, "%f%f%f", &pt[count].x, &pt[count].y, &pt[count].z);
	}
	//double k = 1.0;
	//给dx,dy,dz赋值

	for (int i = 1; i < upperbound; i += di) {
		dx += pt[i].x;
		dy += pt[i].y;
		dz += pt[i].z;
	}dx = (double)dx / upperbound;
	dy = (double)dy / upperbound;
	dz = (double)dz / upperbound;

	dx = pt[630].x;
	dy = pt[630].y;
	dz = pt[630].z;
	/************************************************************************************************/
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);

	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);


	glutInitWindowPosition(100, 100);
	windowWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	windowHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("Lighthouse3D - Assimp Demo");


	//  Callback Registration   
	//glutDisplayFunc(renderScene);

	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);


	glutTimerFunc(1000.0f / 60.0f, myTimer, 0);
	std::thread t1(camTimer);
	//(1000.0f / 15.0f, camTimer, 0);

	//cap=cv::VideoCapture("D:/profiles/心血管比赛相关资料/video_20201103_132337.mp4");
	//cap = cv::VideoCapture("F:/project/5-11-6.mp4");
	//    Mouse and Keyboard Callbacks   
	glutMouseFunc(mouse);//用于注册键盘和鼠标事件发生时的回调函数
	//glutKeyboardFunc(keyFunc); //血管部分的键盘函数
	glutKeyboardFunc(processKeys);


	glutMotionFunc(motion);//注册鼠标移动事件的回调函数，人机交互处理
	glutIdleFunc(&renderScene);//在键盘按下时作某事

	//    Init GLEW
	//glewExperimental = GL_TRUE;
	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 not supported\n");
		return(1);
	}

	loadModelFile((string)"modelToMarker.txt");


	//  Init the app (load model and textures) and OpenGL
	if (!init())
		printf("Could not Load the Model\n");

	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));


	// return from main loop
	//glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	program = setupShaders();
	//setupShaders2D();

	//  GLUT main loop
	glutMainLoop();

	delete[] pt;
	fclose(fp);
	fp = NULL;

	// cleaning up
	textureIdMap.clear();
	
	// clear myMeshes stuff  清除myMeshes内容
	map<int, MyModel>::iterator it;
	for (it = models.begin(); it != models.end(); it++)
	{
		MyModel currentModel = it->second;

		for (unsigned int j = 0; j < currentModel.myMeshes.size(); ++j) {
			glDeleteVertexArrays(1, &(currentModel.myMeshes[j].vao));
			glDeleteTextures(1, &(currentModel.myMeshes[j].texIndex));
			glDeleteBuffers(1, &(currentModel.myMeshes[j].uniformBlockIndex));
		}
	}
	
	// delete buffers
	glDeleteBuffers(1, &matricesUniBuffer);

	exit(0);

}
