#ifndef IMAGE_ITERATOR_CV_ZZZ_H
#define IMAGE_ITERATOR_CV_ZZZ_H
#include <opencv2/opencv.hpp>
#include <vector>

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
    std::vector<std::string> imagePaths;
    int currentIndex;
};

#endif
