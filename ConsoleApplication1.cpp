// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
using namespace cv;
using namespace std;

int thre = 10;
Mat src;
//void trackBar(int, void*); 
int brisk(string name);
void test();
//void fast();

//*******************************************************************   main   *********************
int main(int argc, char** argv)
{
	//fast();
	string tst = "name (3).ppm";
	//brisk(tst);
	test();
	//waitKey();
	return 0;
}

//*******************************************************************   brisk   *********************
int brisk(string name) {

	Mat src1 = imread(name, IMREAD_GRAYSCALE);//input img1 //blocs_pc.pgm
	cout << "testing img " << name << "......"<<endl;
	if (src1.empty()) { cout << "erreur empty"<<endl; return-1; }
	Mat src2 = imread("fresque.ppm", IMREAD_GRAYSCALE);//input img2//blocs.pgm
	Ptr<BRISK> brisk = BRISK::create();//brisk
	vector<KeyPoint>keypoints1, keypoints2,keypointsnew1,keypointsnew2;
	KeyPoint k1;
	keypointsnew1.push_back(k1); keypointsnew2.push_back(k1);
	vector<DMatch>matches2;
	BFMatcher matcher2;
	Mat match_img2;
	drawMatches(src1, keypointsnew1, src2, keypointsnew2, matches2, match_img2);
	//imshow("match_img2", match_img2);


	Mat descriptors1, descriptors2;
	//brisk->detectAndCompute(src1, Mat(), keypoints1, descriptors1);
	//brisk->detectAndCompute(src2, Mat(), keypoints2, descriptors2);
	//*******************************************************************   keypoints   *********************
	Mat dst1 = src1.clone();
	Mat dst2 = src2.clone();
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(thre); //FAST
	detector->detect(src1, keypoints1);
	if (keypoints1.size()<=4) {
		return -1;
		cout << "erreur : keypoints==0" << endl;
	}
	drawKeypoints(dst1, keypoints1, dst1, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	detector->detect(src2, keypoints2);
	drawKeypoints(dst2, keypoints2, dst2, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);

	//imshow("keypoint1", dst1);
	//imwrite("keypoint1.png", dst1);
	//imshow("keypoint2", dst2);
	//imwrite("keypoint2.png", dst2);

	brisk->compute(src1, keypoints1, descriptors1);//brisk
	brisk->compute(src2, keypoints2, descriptors2);
	//namedWindow("descriptors1", WINDOW_AUTOSIZE);
	//imshow("descriptors1", descriptors1);
	//namedWindow("descriptors2", WINDOW_AUTOSIZE);
	//imshow("descriptors2", descriptors2);
	//*******************************************************************   match images   *********************
	BFMatcher matcher;
	vector<DMatch>matches;
	matcher.match(descriptors1, descriptors2, matches);
	Mat match_img;
	Mat match_img_sans_points;
	drawMatches(src1, keypoints1, src2, keypoints2, matches, match_img);
	//imshow("match_img", match_img);
	//imwrite("match_img.png", match_img);
	//*******************************************************************   min distance   *********************
	double minDist = 1000;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;//min distance
		}
	}
	cout << "min distance is " << minDist << endl;

	//*******************************************************************   good matches   *********************
	vector<DMatch>goodMatches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < max(1.8*minDist, 0.02))
		{
			goodMatches.push_back(matches[i]);
		}
	}
	if (goodMatches.size() <=3) {
		cout << "erreur match" << endl;
		return -1;
	}
	Mat good_match_img;
	drawMatches(src1, keypoints1, src2, keypoints2, goodMatches, good_match_img/*, Scalar::all(-1), Scalar::all(-1), vector<char>(), 2*/);
	//imshow("goodMatch", good_match_img);


	//*******************************************************************   print images (result)   *********************
	vector<Point2f>src1GoodPoints, src2GoodPoints;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		src1GoodPoints.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		src2GoodPoints.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}
	Mat P = findHomography(src1GoodPoints, src2GoodPoints, RANSAC);
	vector<Point2f> src1corner(4);
	vector<Point2f> src2corner(4);
	src1corner[0] = Point(0, 0);
	src1corner[1] = Point(src1.cols, 0);
	src1corner[2] = Point(src1.cols, src1.rows);
	src1corner[3] = Point(0, src1.rows);
	perspectiveTransform(src1corner, src2corner, P);
	line(match_img2, src2corner[0] + Point2f(src1.cols, 0), src2corner[1] + Point2f(src1.cols, 0), Scalar(0, 0, 255), 2);
	line(match_img2, src2corner[1] + Point2f(src1.cols, 0), src2corner[2] + Point2f(src1.cols, 0), Scalar(0, 0, 255), 2);
	line(match_img2, src2corner[2] + Point2f(src1.cols, 0), src2corner[3] + Point2f(src1.cols, 0), Scalar(0, 0, 255), 2);
	line(match_img2, src2corner[3] + Point2f(src1.cols, 0), src2corner[0] + Point2f(src1.cols, 0), Scalar(0, 0, 255), 2);
	//imshow("result", match_img2);
	//imwrite("good_match_img.png", match_img2);
	return 0;
}

//*******************************************************************   Threshold track   *********************
//void trackBar(int, void*)
//{
//	std::vector<KeyPoint> keypoints;
//	Mat dst = src.clone();
//	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(thre);
//	detector->detect(src, keypoints);
//	drawKeypoints(dst, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);
//	imshow("output", dst);
//	cout << "show output" << endl;
//	imwrite("beta.png", dst);
//	int comp = 0;
//	KeyPoint tab[255];
//	for(int i =0;i<keypoints.size();i++)
//	{
//		cout << "| size=" << keypoints[i].size << "\t angle=" << keypoints[i].angle<<"\t | \t at position x= "<<keypoints[i].pt.x<<"\t y="<< keypoints[i].pt.y <<" |"<< endl;
//		comp++; 
//		tab[i] = keypoints[i];
//	}
//	//if (keypoints.empty())
//		cout <<comp<< "  KeyPoints in total, done." << endl;
//}

//*******************************************************************   fast   *********************
//void fast() {
//	src = imread("blocs.pgm", 0);
//	if (src.empty())
//	{
//		cout << "can not load image \n" << endl;
//		return;
//	}
//	namedWindow("input", WINDOW_AUTOSIZE);
//	imshow("input", src);
//	cout << "show input" << endl;
//
//	createTrackbar("threshould", "output", &thre, 255, trackBar);
//}

void test() {
	ifstream myfile("name.txt");
	string name;
	int nb=0;
	while (getline(myfile, name))
	{
		cout << name << endl;
		if (brisk(name) == 0)
			nb++;
		cout << nb << endl;
	}
	cout << nb << " imgs marche" << endl;
	return;
}