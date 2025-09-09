#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerStatic.hpp"

#define STATIC_VERTEX_PATH "shaders/vertexShader.vs"

/// @brief Engine in which particle are initlized in a static shape (a sphere and a cube in this case)
class EngineStatic : public AEngine
{
    private:
        /// @brief The cuda worker used for the calculation
        WorkerStatic _worker;

    public:
        EngineStatic();
        EngineStatic(int particleQuantity);
        EngineStatic(const EngineStatic &other);
        virtual ~EngineStatic();

        EngineStatic &operator=(const EngineStatic &other);
        
        void resetCube();

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();
};