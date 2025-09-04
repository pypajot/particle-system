#include <vector>

class FPSCounter
{
    private:
        std::vector<float> _frameTimes;
        int _currentFrame;
        int _calculatePeriod;

    public:
        FPSCounter();
        FPSCounter(int period, float currentTime);
        FPSCounter(const FPSCounter &other);
        ~FPSCounter();

        FPSCounter &operator=(const FPSCounter &other);

        void addFrame(float time);
        int getFPS() const;
        int getFrame() const;

};