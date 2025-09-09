#pragma once

#include "AEngine.hpp"
#include "Worker/WorkerGen.hpp"

#define BASE_PPF 10000
#define MAX_PPF 20000
#define MIN_PPF 1000
#define BASE_TTL 300

#define GEN_VERTEX_PATH "shaders/vertexShaderGen.vs"

/// @brief Engine class in which particle are initialized from a generator
class EngineGen : public AEngine
{
    private:
        /// @brief The cuda worker that will perform the calculation
        WorkerGen _worker;
        /// @brief The number of  particle generated per frame
        int _particlePerFrame;
        /// @brief The number of frame generated particle will live
        int _timeToLive;

     public:
        /// @brief The status of the generator
        bool generatorOn;

        EngineGen();
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