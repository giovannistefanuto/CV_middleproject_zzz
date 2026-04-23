#ifndef FEATURE_CV_ZZZ_H
#define FEATURE_CV_ZZZ_H
#include <opencv2/highgui.hpp>

class Feature
{
    public:
        Feature(cv::Point point, bool presence = true, bool movement = true);

        cv::Point getPoint() const;

        void setFeaturePresence(bool presence);
        void setFeaturePosition(bool movement);
        bool isFeaturePresent() const;
        bool isFeatureMoving() const;
    private:
        cv::Point point;
        bool presence;
        bool movement;
};

#endif