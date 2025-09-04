#pragma once

#include "Worker/AWorker.hpp"

class vec3;

#define BASE_TTL 300
#define BASE_PPF 10000

class WorkerGen : AWorker
{
    private:
        float maxTtl;
        int currentParticle;
        int particlePerFrame;

    public:
        WorkerGen();
        WorkerGen(GLuint VBO, int particleQuantity, float maxTtl, int particlePerFrame);
        WorkerGen(const WorkerGen &other);
        ~WorkerGen();

        WorkerGen &operator=(const WorkerGen &other);

        void call(vec3 &gravityPos, bool gravityOn);
        void init();

};