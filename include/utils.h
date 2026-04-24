#ifndef UTILS_CV_ZZZ_H
#define UTILS_CV_ZZZ_H
#include <string>
#include <vector>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

std::vector<cv::Point> extract_ground_truth(const std::string& path);
std::string getLastPart(const std::string& path);
float evaluate_mIoU(std::vector<cv::Point>&, std::vector<cv::Point>&);
void detectSIFTPoints(const cv::Mat&, cv::Ptr<cv::SIFT>&, std::vector<cv::Point2f>&);
bool computeBoundingBoxFromPoints(const std::vector<cv::Point2f>&, const cv::Size&, cv::Rect&);

int featureFilter(std::vector<cv::Point2f>& newPoints, std::vector<cv::Point2f>& activePoints, std::vector<uchar>& active, std::vector<cv::Point2f>& savedPoints, bool verbose);
bool saveFrame(const std::string& folder, const cv::Mat& frame, const cv::Rect& box, const std::vector<cv::Point2f>& savedPoints, int frameCounter, bool showFeatures);
bool keepDebugOutput(const std::string& path);

//static void drawBoundingBoxFromPoints(cv::Mat& image, const std::vector<cv::Point2f>& points);

#endif // UTILS_DI_PROVA_HPP