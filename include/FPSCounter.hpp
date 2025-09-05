#include <vector>

/// @brief Class that can calculate the mean frame per seconds over a number of frame passed as argument of the constructor.
/// @note Can use system time if no time argument is passed with the constructor.
class FPSCounter
{
    private:
        /// @brief The array storing the frame times
        std::vector<double> _frameTimes;
        /// @brief The current frame modulo the frame period + 1
        int _currentFrame;
        /// @brief The frame period over which the mean fps will be calculated
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