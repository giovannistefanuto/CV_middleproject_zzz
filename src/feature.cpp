#include "feature.h"

Feature::Feature(cv::Point point, bool presence, bool movement)
{
	this->point = point;
	this->presence = presence;
	this->movement = movement;
}

cv::Point Feature::getPoint() const
{
	return point;
}

void Feature::setFeaturePresence(bool presence)
{
	this->presence = presence; 
}

void Feature::setFeaturePosition(bool movement)
{
	this->movement = movement;
}

bool Feature::isFeaturePresent() const
{
	return presence;
}

bool Feature::isFeatureMoving() const
{
	return movement;
}
