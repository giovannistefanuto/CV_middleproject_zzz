#ifndef UTILS_DI_PROVA_HPP
#define UTILS_DI_PROVA_HPP
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

float evaluate_mIoU(std::vector<cv::Point>& predicted_points, std::vector<cv::Point>& ground_truth_points, cv::Mat& output_image=cv::noArray());

#endif // UTILS_DI_PROVA_HPP