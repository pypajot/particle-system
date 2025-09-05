#include <vector>

/// @brief Class that can calculate the mean fps over a period of time passed as argument of the constructor.
/// @note Can use system time if no time argument is passed with the constructor.
class FPSCounter
{
    private:
        std::vector<double> _frameTimes;
        int _currentFrame;
        int _calculatePeriod;

    public:
        FPSCounter();
        FPSCounter(int period);
        FPSCounter(int period, double currentTime);
        FPSCounter(const FPSCounter &other);
        ~FPSCounter();

        FPSCounter &operator=(const FPSCounter &other);

        void addFrame();
        void addFrame(double time);
        int getFPS() const;
        int getFrame() const;

};