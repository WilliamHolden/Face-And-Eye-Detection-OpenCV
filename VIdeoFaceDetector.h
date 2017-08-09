#pragma once

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>

using namespace cv;
using namespace std;

class VideoFaceDetector
{
public:
	VideoFaceDetector(const std::string cascadeFilePath, VideoCapture &videoCapture);
	~VideoFaceDetector();

	Point               getFrameAndDetect(Mat &frame);
	Point               operator >> (Mat &frame);
	void                    setVideoCapture(VideoCapture &videoCapture);
	VideoCapture*       videoCapture() const;
	void                    setFaceCascade(const std::string cascadeFilePath);
	CascadeClassifier*  faceCascade() const;
	void                    setResizedWidth(const int width);
	int                     resizedWidth() const;
	bool					isFaceFound() const;
	Rect                face() const;
	Point               facePosition() const;
	void                    setTemplateMatchingMaxDuration(const double s);
	double                  templateMatchingMaxDuration() const;

private:
	static const double     TICK_FREQUENCY;

	VideoCapture*       m_videoCapture = NULL;
	CascadeClassifier*  m_faceCascade = NULL;
	std::vector< Rect>   m_allFaces;
	Rect                m_trackedFace;
	Rect                m_faceRoi;
	Mat                 m_faceTemplate;
	Mat                 m_matchingResult;
	bool                    m_templateMatchingRunning = false;
	int64                   m_templateMatchingStartTime = 0;
	int64                   m_templateMatchingCurrentTime = 0;
	bool                    m_foundFace = false;
	double                  m_scale;
	int                     m_resizedWidth = 320;
	Point               m_facePosition;
	double                  m_templateMatchingMaxDuration = 3;

	Rect    doubleRectSize(const  Rect &inputRect, const  Rect &frameSize) const;
	Rect    biggestFace(std::vector< Rect> &faces) const;
	Point   centerOfRect(const  Rect &rect) const;
	Mat     getFaceTemplate(const  Mat &frame, Rect face);
	void        detectFaceAllSizes(const  Mat &frame);
	void        detectFaceAroundRoi(const  Mat &frame);
	void        detectFacesTemplateMatching(const  Mat &frame);
};