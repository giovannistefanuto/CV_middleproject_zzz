#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <utils.h>
#include <fstream>
#include <algorithm>
#include <imageIterator.h>

bool processCategory(const std::string& inputFolder)
{
    const bool showSavedFeatures = true;
    bool logFeatureMotion=keepDebugOutput(inputFolder);
    
    // Iteratore immagini della categoria (supporta PNG/JPG ordinati).
    ImageIterator iterator(inputFolder);
    if (!iterator.hasNext()) {
        std::cerr << "Errore: nessuna immagine PNG/JPG trovata in " << inputFolder << std::endl;
        return false;
    }

    int aliveFeatures, numFrames=15;
    std::string category= getLastPathPart(inputFolder);
    std::vector<cv::Mat> frames(numFrames);
    std::vector<float> motions, err;
    std::vector<uchar> status;
    std::vector<cv::Point2f> referencePoints, movedPoints, printablePoints;
    std::vector<cv::Point> realPoints = extract_ground_truth(category);

    cv::Rect box;
    cv::Mat referenceGray, actualGray;
    cv::Point box_point_1,box_point_2;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(60000);

    for(int i=0;iterator.hasNext();i++)
    {
        iterator.next(frames[i%numFrames]);
        if(i%numFrames==0)
        {
            cv::cvtColor(frames[0], referenceGray, cv::COLOR_BGR2GRAY);
            referencePoints.clear();
            detectSIFTPoints(referenceGray, sift, referencePoints);
            motions = std::vector<float>(referencePoints.size());
            continue;
        }
        cv::cvtColor(frames[i%numFrames], actualGray, cv::COLOR_BGR2GRAY);
        cv::calcOpticalFlowPyrLK(referenceGray, actualGray, referencePoints, movedPoints, status, err);
        accumulateMotion(movedPoints,referencePoints,status,motions);
        if((i+1)%numFrames==0 || !iterator.hasNext())
        //if(!iterator.hasNext())
        {
            aliveFeatures = featureFilter(referencePoints,printablePoints,motions);
            computeBoundingBoxFromPoints(printablePoints, frames[0].size(), box);
            box_point_1= cv::Point(box.x,box.y);
            box_point_2= cv::Point(box.x+box.width,box.y+box.height);
            std::vector<cv::Point> boxPoints{box_point_1,box_point_2};

            if(i==numFrames-1)
            {
                float score=evaluate_mIoU(boxPoints,realPoints);
                std::cout<<score<<std::endl;
            }
            bool result=true;
            for(int j=0;j<numFrames && j<i;j++)
            {   
                //result &= saveFrame(inputFolder,frames[j],box,printablePoints,j+1,showSavedFeatures);
                result &= saveFrame(inputFolder,frames[j],box,printablePoints,i-(numFrames-j)+1,showSavedFeatures);
            }
            if(!result)
            {
                return false;
            }
        }
    }
    return true;
}

std::vector<cv::Point> extract_ground_truth(const std::string& path)
{
    std::ifstream file("../dataset/labels/"+path+"/0000.txt");
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

std::string getLastPathPart(const std::string& path)
{
    size_t pos=path.find_last_of("/\\");
    if(pos==std::string::npos)
        return path;
    return path.substr(pos+1);
}

float evaluate_mIoU(const std::vector<cv::Point>& predicted_points, const std::vector<cv::Point>& ground_truth_points) 
{
    // Controllo di sicurezza base
    if (predicted_points.size() < 2 || ground_truth_points.size() < 2) {
        return 0.0f;
    }

    // Trova i minimi e massimi per entrambi i box
    int min_pred_x = std::min(predicted_points[0].x, predicted_points[1].x);
    int max_pred_x = std::max(predicted_points[0].x, predicted_points[1].x);
    int min_pred_y = std::min(predicted_points[0].y, predicted_points[1].y);
    int max_pred_y = std::max(predicted_points[0].y, predicted_points[1].y);

    int min_ground_x = std::min(ground_truth_points[0].x, ground_truth_points[1].x);
    int max_ground_x = std::max(ground_truth_points[0].x, ground_truth_points[1].x);
    int min_ground_y = std::min(ground_truth_points[0].y, ground_truth_points[1].y);
    int max_ground_y = std::max(ground_truth_points[0].y, ground_truth_points[1].y);

    // Calcola le coordinate del rettangolo di intersezione
    int inter_min_x = std::max(min_pred_x, min_ground_x);
    int inter_min_y = std::max(min_pred_y, min_ground_y);
    int inter_max_x = std::min(max_pred_x, max_ground_x);
    int inter_max_y = std::min(max_pred_y, max_ground_y);

    // Calcola l'area di intersezione (se non si sovrappongono, w o h saranno 0)
    float inter_w = std::max(0, inter_max_x - inter_min_x);
    float inter_h = std::max(0, inter_max_y - inter_min_y);
    float inter_area = inter_w * inter_h;

    // Calcola le aree dei singoli box
    float pred_area = (max_pred_x - min_pred_x) * (max_pred_y - min_pred_y);
    float ground_area = (max_ground_x - min_ground_x) * (max_ground_y - min_ground_y);

    // Calcola la vera area di unione
    float union_area = pred_area + ground_area - inter_area;

    // Previene la divisione per zero
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
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

void accumulateMotion(const std::vector<cv::Point2f>& newPoints, const std::vector<cv::Point2f>& activePoints, const std::vector<uchar>& active, std::vector<float>& allMotions)
{
    //float minMovement = 0.70f;
    const float maxMovement = 100.0f;
    float motion, max=0;

    for (size_t i = 0; i < newPoints.size(); ++i)
    {
        if (!active[i])
        {
            allMotions.push_back(0.0f);
            continue;
        }

        float dx = newPoints[i].x - activePoints[i].x;
        float dy = newPoints[i].y - activePoints[i].y;
        motion = std::sqrt(dx * dx + dy * dy);
        
        if(motion<maxMovement){
            allMotions[i]+=motion;
        }
    }
}

int featureFilter(const std::vector<cv::Point2f>& activePoints, std::vector<cv::Point2f>& savedPoints, const std::vector<float>& allMotions)
{

    float minMovement;

    float sigma;
    float meanMovement=0.0f;
    for(size_t i=0; i<allMotions.size(); i++)
    {
        meanMovement+=allMotions[i];
    }
    meanMovement/=allMotions.size();

    float sq_sum = 0.0f;
    for (float m : allMotions) {
        sq_sum += (m - meanMovement) * (m - meanMovement);
    }

    minMovement = meanMovement+1*std::sqrt(std::sqrt(sq_sum));
    int survived=0;
    savedPoints.clear();

    for (size_t i = 0; i < activePoints.size(); ++i){
        if(allMotions[i]>minMovement)
        {
            savedPoints.push_back(activePoints[i]);
            ++survived;
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

    bool logFeatureMotion = true;
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
