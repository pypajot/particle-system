#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerStatic.hpp"

#define STATIC_VERTEX_PATH "shaders/vertexShader.vs"

class EngineStatic : public AEngine
{
    private:
        WorkerStatic _worker;

    public:
        EngineStatic(int particleQuantity);
        EngineStatic(const EngineStatic &other);
        virtual ~EngineStatic();

        EngineStatic &operator=(const EngineStatic &other);
        
        void resetCube();

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();
};