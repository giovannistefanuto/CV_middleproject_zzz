class featureList
{
public:
    featureList();
    void addFeature(cv::Point feature);
    void addFeature(Feature feature);
    std::vector<Feature> getFeatures() const;
    std::vector<cv::Point> getFeaturePoints() const;
    Feature& getFeature(int index) const;
private:
    std::vector<Feature> isPresentandMoved;
    std::vector<Feature> isPresentandNotMoved;
    std::vector<Feature> isNotPresentandMoved;
};

