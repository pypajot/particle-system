#pragma once

#include "AEngine.hpp"

class EngineStatic : public AEngine
{
    public:
        EngineStatic(int particleQuantity);

        void initCube();
        void initSphere();

        void reset();
        void run();
};