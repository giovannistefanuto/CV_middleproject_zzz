#ifndef UTILS_CV_ZZZ_H
#define UTILS_CV_ZZZ_H
#include <string>
#include <vector>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/opencv.hpp>

std::vector<cv::Point> extract_ground_truth(const std::string& path);
std::string getLastPathPart(const std::string& path);
float evaluate_mIoU(const std::vector<cv::Point>& predicted_points, const std::vector<cv::Point>& ground_truth_points);
void detectSIFTPoints(const cv::Mat& gray, cv::Ptr<cv::SIFT>& sift, std::vector<cv::Point2f>& points);
bool computeBoundingBoxFromPoints(const std::vector<cv::Point2f>& points, const cv::Size& imageSize, cv::Rect& box);
bool processCategory(const std::string& inputFolder);


void accumulateMotion(const std::vector<cv::Point2f>& newPoints, const std::vector<cv::Point2f>& activePoints, const std::vector<uchar>& active, std::vector<float>& allMotions);
int featureFilter(const std::vector<cv::Point2f>& activePoints, std::vector<cv::Point2f>& savedPoints, const std::vector<float>& allMotions);
bool saveFrame(const std::string& folder, const cv::Mat& frame, const cv::Rect& box, const std::vector<cv::Point2f>& savedPoints, int frameCounter, bool showFeatures);
bool keepDebugOutput(const std::string& path);

#endif // UTILS_DI_PROVA_HPP