#include "featureList.h"

#include <stdexcept>

featureList::featureList()
{
	// Costruttore vuoto
}

void featureList::addFeature(cv::Point featurePoint)
{
	// Creo una Feature a partire dal punto e la aggiungo
	Feature newFeature(featurePoint);
	addFeature(newFeature);
}

void featureList::addFeature(Feature feature)
{
	// Di default considero la feature presente e mossa
	isPresentandMoved.push_back(feature);
}

std::vector<Feature> featureList::getFeatures() const
{
	std::vector<Feature> all;

	for (size_t i = 0; i < isPresentandMoved.size(); i++)
	{
		all.push_back(isPresentandMoved[i]);
	}

	for (size_t i = 0; i < isPresentandNotMoved.size(); i++)
	{
		all.push_back(isPresentandNotMoved[i]);
	}

	for (size_t i = 0; i < isNotPresentandMoved.size(); i++)
	{
		all.push_back(isNotPresentandMoved[i]);
	}

	return all;
}

std::vector<cv::Point> featureList::getFeaturePoints() const
{
	std::vector<cv::Point> points;
	std::vector<Feature> all = getFeatures();

	for (size_t i = 0; i < all.size(); i++)
	{
		points.push_back(all[i].getPoint());
	}

	return points;
}

Feature& featureList::getFeature(int index) const
{
	if (index < 0)
	{
		throw std::out_of_range("Index negativo");
	}

	int current = index;

	if (current < static_cast<int>(isPresentandMoved.size()))
	{
		return const_cast<Feature&>(isPresentandMoved[current]);
	}
	current -= static_cast<int>(isPresentandMoved.size());

	if (current < static_cast<int>(isPresentandNotMoved.size()))
	{
		return const_cast<Feature&>(isPresentandNotMoved[current]);
	}
	current -= static_cast<int>(isPresentandNotMoved.size());

	if (current < static_cast<int>(isNotPresentandMoved.size()))
	{
		return const_cast<Feature&>(isNotPresentandMoved[current]);
	}

	throw std::out_of_range("Index fuori range");
}
