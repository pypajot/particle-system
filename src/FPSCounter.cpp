#include <chrono>

#include "FPSCounter.hpp"

double getSecondsFromTimepoint(std::chrono::_V2::steady_clock::time_point point)
{
    return std::chrono::duration_cast<std::chrono::seconds>(point.time_since_epoch()).count();
}

/// @brief Construct a FPSCounter object with using system time and a fps period of 60
FPSCounter::FPSCounter() : FPSCounter(60, 0)
{
}

/// @brief Construct a FPSCounter with system time
/// @param period The fps period over wich the mean time will be calculated
FPSCounter::FPSCounter(int period)
{
    _currentFrame = 0;
    _calculatePeriod = period;
    auto time = std::chrono::steady_clock::now();
    _frameTimes = std::vector<double>(period + 1, getSecondsFromTimepoint(time));
}

/// @brief Construct a FPSCounter with a custom time system 
/// @param period The fps period over wich the mean time will be calculated
/// @param currentTime The base of the system time used to calculate the fps 
FPSCounter::FPSCounter(int period, double currentTime)
{
    _currentFrame = 0;
    _calculatePeriod = period;
    _frameTimes = std::vector<double>(period + 1, currentTime);
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

/// @brief Add a frame to time using system time
/// @warning Need to be used with the constructor using system time and itself, or else the result may be undefined
void FPSCounter::addFrame()
{
    auto time = std::chrono::steady_clock::now();
    _frameTimes[_currentFrame] = getSecondsFromTimepoint(time);
    _currentFrame = _currentFrame + 1 % (_calculatePeriod + 1);
}

/// @brief Add a frame to time using a custom time system
/// @param time Need to be used with the constructor using a custom time system and itself, or else the result may be undefined
void FPSCounter::addFrame(double time)
{
    _frameTimes[_currentFrame] = time;
    _currentFrame = _currentFrame + 1 % (_calculatePeriod + 1);
}

/// @brief Calculate the mean fps over the fps period
/// @return The frame per second over the last period
int FPSCounter::getFPS() const
{
    return _calculatePeriod / (_frameTimes[_currentFrame + 1 % (_calculatePeriod + 1)] - _frameTimes[_currentFrame] + __FLT_EPSILON__);
}

/// @brief Get the current frame in the frame period time frame
/// @return Return the current frame modulo the time period + 1 
int FPSCounter::getFrame() const
{
    return _currentFrame;
}

