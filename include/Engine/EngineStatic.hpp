#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerStatic.hpp"


class EngineStatic : public AEngine
{
    private:
        WorkerGen gravity;
        
    public:
        EngineStatic(int particleQuantity);

        void initCube();
        void initSphere();

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();
};