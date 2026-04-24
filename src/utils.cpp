#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <utils.h>
#include <fstream>
#include <algorithm>


std::vector<cv::Point> extract_ground_truth(const std::string& path){
    std::ifstream file("dataset/labels/"+path+"/0000.txt");
    int temp;
    cv::Point p1,p2;

    // Read up to 4 integers as long as the file stream is healthy
    file >> temp;
    p1.x=temp;
    file >> temp;
    p1.y=temp;
    file >> temp;
    p2.x=temp;
    file >> temp;
    p2.y=temp;

    std::vector<cv::Point> ground_truth{p1,p2};
    return ground_truth;
}

std::string getLastPart(const std::string& path){
    size_t pos=path.find_last_of("/\\");
    if(pos==std::string::npos)
        return path;
    return path.substr(pos+1);
}

float evaluate_mIoU(std::vector<cv::Point>& predicted_points, std::vector<cv::Point>& ground_truth_points){
    std::vector<int> x_s;
    std::vector<int> y_s;

    for(int i=0;i<2;i++){
        x_s.push_back(predicted_points[i].x);
        x_s.push_back(ground_truth_points[i].x);

        y_s.push_back(predicted_points[i].y);
        y_s.push_back(ground_truth_points[i].y);
    }

    std::sort(x_s.begin(),x_s.end());
    std::sort(y_s.begin(),y_s.end());

    float inter_w = std::max(0, x_s[2] - y_s[1]);
    float inter_h = std::max(0, y_s[2] - y_s[1]);
    float inter_area = inter_w * inter_h;

    int min_pred_x=std::min(predicted_points[0].x,predicted_points[1].x);
    int min_pred_y=std::min(predicted_points[0].y,predicted_points[1].y);
    int min_ground_x=std::min(ground_truth_points[0].x,ground_truth_points[1].x);
    int min_ground_y=std::min(ground_truth_points[0].y,ground_truth_points[1].y);

    int max_pred_x=std::max(predicted_points[0].x,predicted_points[1].x);
    int max_pred_y=std::max(predicted_points[0].y,predicted_points[1].y);
    int max_ground_x=std::max(ground_truth_points[0].x,ground_truth_points[1].x);
    int max_ground_y=std::max(ground_truth_points[0].y,ground_truth_points[1].y);

    int union_area=(max_pred_x-min_pred_x)*(max_pred_y-min_pred_y)+(max_ground_x-min_ground_x)*(max_ground_y-min_ground_y);
    float tot_area=union_area-inter_area;

    std::cout<<inter_area<<" "<<union_area<<" "<<tot_area<<std::endl;

    return inter_area/tot_area;
}

void detectSIFTPoints(const cv::Mat& gray, cv::Ptr<cv::SIFT>& sift, std::vector<cv::Point2f>& points)
{
    // Estrae keypoint SIFT e li converte in punti 2D tracciabili con Optical Flow.
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        points.push_back(keypoints[i].pt);
    }
}

bool computeBoundingBoxFromPoints(const std::vector<cv::Point2f>& points, const cv::Size& imageSize, cv::Rect& box)
{
    // Calcola bbox axis-aligned e la limita ai bordi immagine.
    if (points.empty())
    {
        return false;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (size_t i = 1; i < points.size(); ++i)
    {
        minX = std::min(minX, points[i].x);
        maxX = std::max(maxX, points[i].x);
        minY = std::min(minY, points[i].y);
        maxY = std::max(maxY, points[i].y);
    }

    int x1 = std::max(0, cvRound(minX));
    int y1 = std::max(0, cvRound(minY));
    int x2 = std::min(imageSize.width - 1, cvRound(maxX));
    int y2 = std::min(imageSize.height - 1, cvRound(maxY));

    if (x2 <= x1 || y2 <= y1)
    {
        return false;
    }

    box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    return true;
}

int featureFilter(std::vector<cv::Point2f>& newPoints, std::vector<cv::Point2f>& activePoints, std::vector<uchar>& active, std::vector<cv::Point2f>& savedPoints, bool verbose=false){
    //float minMovement = 0.70f;
    const float maxMovement = 100.0f;
    int survived=0;
    std::vector<float> allMotions;
    float motion, max=0;

    for (size_t i = 0; i < newPoints.size(); ++i)
    {
        if (!active[i])
        {
            continue;
        }

        float dx = newPoints[i].x - activePoints[i].x;
        float dy = newPoints[i].y - activePoints[i].y;
        motion = std::sqrt(dx * dx + dy * dy);
        allMotions.push_back(motion);

        if(motion>max && motion<maxMovement){
            max=motion;
        }
    }

    float minMovement=2*max/5;

    for (size_t i = 0; i < newPoints.size(); ++i){
        if (allMotions[i] > minMovement && allMotions[i] < maxMovement)
        {
            // La box del frame 0 deve usare punti del frame 0, non del frame 1.
            savedPoints.push_back(activePoints[i]);
            ++survived;

            if (verbose)
            {
                std::cout << "[DEBUG][Frame 0->1] feature " << i
                          << " prev=(" << activePoints[i].x << "," << activePoints[i].y << ")"
                          << " next=(" << newPoints[i].x << "," << newPoints[i].y << ")"
                          << " motion=" << motion << std::endl;
            }
        }
    }

    return survived;
}

bool saveFrame(const std::string& folder, const cv::Mat& frame, const cv::Rect& box, const std::vector<cv::Point2f>& savedPoints, int frameCounter, bool showSavedFeatures=false){
    // Ogni categoria salva output in una cartella dedicata accanto all'input.
    const std::string outputFolder = folder + "_annotate";

    if (!cv::utils::fs::exists(outputFolder)) {
        cv::utils::fs::createDirectories(outputFolder);
    }

    cv::Mat output = frame.clone();
    cv::rectangle(output, box, cv::Scalar(0, 255, 0), 2);
    if (showSavedFeatures)
    {
        // Visualizza sul frame corrente le feature mantenute dopo i filtri.
        for (size_t i = 0; i < savedPoints.size(); ++i)
        {
            cv::circle(output, savedPoints[i], 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    std::string outputPath = outputFolder + cv::format("/frame_%04d.png", frameCounter);
    if (!cv::imwrite(outputPath, output)) {
        std::cerr << "Errore: impossibile salvare " << outputPath << std::endl;
        return false;
    }
    return true;
}

bool keepDebugOutput(const std::string& path){
    // Flag debug per categoria: abilita/disabilita i log dettagliati del movimento feature.
    const bool logBird = false;
    const bool logCar = false;
    const bool logFrog = false;
    const bool logSheep = false;
    const bool logSquirrel = false;

    bool logFeatureMotion = false;
    if (path.find("bird") != std::string::npos)
    {
        logFeatureMotion = logBird;
    }
    else if (path.find("car") != std::string::npos)
    {
        logFeatureMotion = logCar;
    }
    else if (path.find("frog") != std::string::npos)
    {
        logFeatureMotion = logFrog;
    }
    else if (path.find("sheep") != std::string::npos)
    {
        logFeatureMotion = logSheep;
    }
    else if (path.find("squirrel") != std::string::npos)
    {
        logFeatureMotion = logSquirrel;
    }

    return logFeatureMotion;
}

// TODO: are those used??

/*
static void drawBoundingBoxFromPoints(cv::Mat& image, const std::vector<cv::Point2f>& points)
{
    if (points.empty())
    {
        return;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (size_t i = 1; i < points.size(); ++i)
    {
        minX = std::min(minX, points[i].x);
        maxX = std::max(maxX, points[i].x);
        minY = std::min(minY, points[i].y);
        maxY = std::max(maxY, points[i].y);
    }

    cv::Rect box(
        cv::Point(static_cast<int>(minX), static_cast<int>(minY)),
        cv::Point(static_cast<int>(maxX), static_cast<int>(maxY))
    );

    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
}
*/