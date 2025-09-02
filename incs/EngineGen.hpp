#pragma once

#include "AEngine.hpp"

class EngineGen : public AEngine
{
    public:
        const float timeToLive = 300.0f;
        bool generatorOn;
        int particlePerFrame;
        int currentParticle;

        EngineGen(int particleQuantity);

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();
};