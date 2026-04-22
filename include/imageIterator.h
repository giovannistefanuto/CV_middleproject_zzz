#ifndef IMAGE_ITERATOR_HPP
#define IMAGE_ITERATOR_HPP
#include <opencv2/opencv.hpp>


class ImageIterator
{
public:
    ImageIterator(const std::string& path);

    void next(cv:: Mat& image);
    bool hasNext();

    std::string getPath() const;
    int getCurrentIndex() const;

private:
    std::string path;
    int currentIndex;
    cv::Mat image;
};

#endif // IMAGE_ITERATOR_HPP

