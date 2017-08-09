#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "VideoFaceDetector.h"
#define PI 3.141
#define COLOR Scalar(255, 255, 255)
using namespace cv;
using namespace std;

const  String    WINDOW_NAME("Camera video");
const  String    CASCADE_FACE_FILE("haarcascade_frontalface_default.xml");
const  String    CASCADE_EYE_FILE("haarcascade_eye.xml");

int detectEyes(vector<Rect> &eyes, Mat &face, CascadeClassifier eyeD)
{
	vector<Rect> temp;
	eyeD.detectMultiScale(face, temp, 1.15f, 3, CASCADE_SCALE_IMAGE, Size(face.rows / 5, face.rows / 5), Size(face.rows * 2 / 3, face.rows * 2 / 3));
	if (temp.size()>1) {
		if (temp.at(0).x > temp.at(1).x) { eyes.push_back(temp.at(1)); eyes.push_back(temp.at(0)); }
		else { eyes.push_back(temp.at(0)); eyes.push_back(temp.at(1)); }
		return 1;
	}
	return 0;
}

float getAngle(Point pt1, Point pt2) {
	Point2f vect;
	vect.x = pt1.x - pt2.x;
	vect.y = pt1.y - pt2.y;
	float angle = atan2(vect.y, vect.x);
	if (vect.x != 0) {
		return angle * (180 / PI);
	}
	return 0;	
}

int main(int argc, char** argv)
{
	VideoCapture camera(0);
	if (!camera.isOpened()) {
		fprintf(stderr, "Error opening cam\n");
		exit(1);
	}

	namedWindow(WINDOW_NAME, CV_WINDOW_NORMAL);

	VideoFaceDetector detector(CASCADE_FACE_FILE, camera);
	CascadeClassifier eyeD;
	eyeD.load(CASCADE_EYE_FILE);
	Mat frame;
	if (eyeD.empty()) {
		fprintf(stderr, "Error loading classifier\n");
		exit(1);
	}

	while (true)
	{
		detector >> frame;

		vector<Rect> eyes;
		if (detector.isFaceFound())
		{
			Rect tempFaceRect = detector.face();
			rectangle(frame, tempFaceRect, Scalar(255, 0, 0));
			Mat extFace(frame, tempFaceRect);
			if (detectEyes(eyes, extFace, eyeD)) {
				eyes[0].x += tempFaceRect.x;
				eyes[0].y += tempFaceRect.y;
				eyes[1].x += tempFaceRect.x;
				eyes[1].y += tempFaceRect.y;
				rectangle(frame, eyes.at(0),  Scalar(255, 0, 0));
				rectangle(frame, eyes.at(1),  Scalar(255, 0, 0));
				float faceAngle = getAngle(Point(eyes[0].x, eyes[0].y), Point(eyes[1].x, eyes[1].y));

				ellipse(frame, Point(eyes[0].x + (eyes[0].width / 2), eyes[0].y + (eyes[0].height / 2)), Size(eyes[0].width / 2, eyes[0].height / 5), faceAngle, 0, 360, COLOR, -1);
				ellipse(frame, Point(eyes[1].x + (eyes[1].width / 2), eyes[1].y + (eyes[1].height / 2)), Size(eyes[1].width / 2, eyes[1].height / 5), faceAngle, 0, 360, COLOR, -1);
			}
		}
		imshow(WINDOW_NAME, frame);
		if (waitKey(10) == 27) break;
	}

	return 0;
}