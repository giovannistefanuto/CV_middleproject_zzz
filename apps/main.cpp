#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>

int main(int argc, char** argv){

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <img1> <img2>" << std::endl;
        return 1;
    }

    cv::Mat img=cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2=cv::imread(argv[2], cv::IMREAD_COLOR);
    if (img.empty() || img2.empty()) {
        std::cerr << "Errore: impossibile leggere una o entrambe le immagini." << std::endl;
        return 1;
    }

    cv::Mat gray;
    cv::Mat gray2;
    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2,gray2,cv::COLOR_BGR2GRAY);

    if (gray.size() != gray2.size()) {
        std::cerr << "Errore: le immagini devono avere la stessa dimensione per Lucas-Kanade." << std::endl;
        return 1;
    }
    if (gray.type() != gray2.type()) {
        std::cerr << "Errore: le immagini devono avere lo stesso tipo per Lucas-Kanade." << std::endl;
        return 1;
    }

    cv::Mat descriptors, output1;
    std::vector<cv::KeyPoint> keypoints1;

    cv::Ptr<cv::SIFT> feature_finder = cv::SIFT::create(40000);
    
    auto inizio= std::chrono::steady_clock::now();
    feature_finder->detectAndCompute(gray,cv::noArray(),keypoints1,descriptors);
    auto fine= std::chrono::steady_clock::now();

    std::chrono::duration<double> durata=(fine - inizio);
    std::cout<<durata.count()<<std::endl;
    
    cv::drawKeypoints(img, keypoints1, output1, cv::Scalar(255,0,0));
    cv::imshow("sift_result_1", output1);

    std::cout<<"trovati "<<keypoints1.size()<<std::endl;

    //Vogliamo usare calcOpticalFlowPyrLK per trovare i punti di interesse che ci sono nella prima immagine e che si trovano anche nella seconda immagine, e vedere se si muovono o no.
    std::vector<cv::Point2f> points1, points2;
    for (const auto& keypoint : keypoints1) {
        points1.push_back(keypoint.pt);
    }
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(gray, gray2, points1, points2, status, err);
    cv::Mat output2 = img2.clone();
    for (size_t i = 0; i < points1.size(); ++i)
    {
        if (status[i]) {
            cv::line(output2, points1[i], points2[i], cv::Scalar(0, 255, 0), 2);
            cv::circle(output2, points2[i], 3, cv::Scalar(0, 0, 255), -1);
        }
    }
    //for per vedere punto 1 immagine 1 e punto 1 immagine 2, da stampare tutti i punti e vedere se si muovono o no, se si muovono allora sono verdi, altrimenti rossi.
    for (size_t i = 0; i < points1.size(); ++i)
    {
        if (status[i]) {
            std::cout << "Punto " << i << ": (" << points1[i].x << ", " << points1[i].y << ") -> (" << points2[i].x << ", " << points2[i].y << ")" << std::endl;
        } else {
            std::cout << "Punto " << i << ": (" << points1[i].x << ", " << points1[i].y << ") -> non trovato nella seconda immagine" << std::endl;
        }
    }
    cv::imshow("sift_result_2", output2);

    

    cv::waitKey(0);
    return 0;
}