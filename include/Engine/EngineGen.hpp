#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerGen.hpp"

class EngineGen : public AEngine
{
    private:
        WorkerGen worker;
        
    public:
        bool generatorOn;

        EngineGen(const EngineGen &other);
        EngineGen(int particleQuantity);
        virtual ~EngineGen();

        EngineGen &operator=(const EngineGen &other);

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();
};