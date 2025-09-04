#include "FPSCounter.hpp"

FPSCounter::FPSCounter()
{
}

FPSCounter::FPSCounter(int period, float currentTime)
{
    _currentFrame = 0;
    _calculatePeriod = period;
    _frameTimes = std::vector<float>(period + 1, currentTime);
}

FPSCounter::FPSCounter(const FPSCounter &other)
{
    _currentFrame = other._currentFrame;
    _calculatePeriod = other._calculatePeriod;
    _frameTimes = other._frameTimes;
}

FPSCounter::~FPSCounter()
{
}

FPSCounter &FPSCounter::operator=(const FPSCounter &other)
{
    if (this == &other)
        return *this;

    _currentFrame = other._currentFrame;
    _calculatePeriod = other._calculatePeriod;
    _frameTimes = other._frameTimes;

    return *this;
}

void FPSCounter::addFrame(float time)
{
    _frameTimes[_currentFrame] = time;
    _currentFrame = _currentFrame + 1 % (_calculatePeriod + 1);
}

int FPSCounter::getFPS() const
{
    return _calculatePeriod / (_frameTimes[_currentFrame + 1 % (_calculatePeriod + 1)] - _frameTimes[_currentFrame] + __FLT_EPSILON__);
}

int FPSCounter::getFrame() const
{
    return _currentFrame;
}

