#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerGen.hpp"

#define BASE_PPF 10000
#define MAX_PPF 20000
#define MIN_PPF 1000

class EngineGen : public AEngine
{
    private:
        WorkerGen worker;
        int particlePerFrame;
        
     public:
        bool generatorOn;

        EngineGen(const EngineGen &other);
        EngineGen(int particleQuantity);
        virtual ~EngineGen();

        EngineGen &operator=(const EngineGen &other);

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void reset();
        void run();

        void ppfUp();
        void ppfDown();
};