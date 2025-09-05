#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerGen.hpp"

#define BASE_PPF 10000
#define MAX_PPF 20000
#define MIN_PPF 1000
#define BASE_TTL 300


class EngineGen : public AEngine
{
    private:
        WorkerGen _worker;
        int _particlePerFrame;

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