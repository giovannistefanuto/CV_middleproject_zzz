#include <imageIterator.h>
#include <opencv2/core/utils/filesystem.hpp>


ImageIterator::ImageIterator(const std::string& path){

    if (!cv::utils::fs::exists(path)) {
        std::cerr << "Errore: cartella input non trovata: " << path << std::endl;
        return;
    }

    this->path = path;
    currentIndex = 0;

    std::vector<std::string> pngFiles;
    std::vector<std::string> jpgFiles;
    std::vector<std::string> jpegFiles;

    cv::glob(this->path + "/*.png", pngFiles, false);
    cv::glob(this->path + "/*.jpg", jpgFiles, false);
    cv::glob(this->path + "/*.jpeg", jpegFiles, false);

    imagePaths.reserve(pngFiles.size() + jpgFiles.size() + jpegFiles.size());
    imagePaths.insert(imagePaths.end(), pngFiles.begin(), pngFiles.end());
    imagePaths.insert(imagePaths.end(), jpgFiles.begin(), jpgFiles.end());
    imagePaths.insert(imagePaths.end(), jpegFiles.begin(), jpegFiles.end());

    std::sort(imagePaths.begin(), imagePaths.end());
}

void ImageIterator::next(cv::Mat& image){
    if(hasNext()){
        image = cv::imread(imagePaths[currentIndex], cv::IMREAD_COLOR);
        ++currentIndex;
    }
}

bool ImageIterator::hasNext(){
    return currentIndex < static_cast<int>(imagePaths.size());
}

std::string ImageIterator::getPath() const{
    return path;
}

int ImageIterator::getCurrentIndex() const{
    return currentIndex;
}