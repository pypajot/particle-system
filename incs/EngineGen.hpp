#pragma once

#include "AEngine.hpp"

class EngineGen : public AEngine
{
    public:
        const int timeToLive = 120;
        bool generatorOn;
        int particlePerFrame;
        int currentParticle;

        EngineGen(int particleQuantity);

        void reset();
        void run();
};