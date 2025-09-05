#pragma once

#include "Worker/AWorker.hpp"

class vec3;

class WorkerGen : AWorker
{
    private:
        int currentParticle;
        
    public:
        WorkerGen();
        WorkerGen(GLuint VBO, int particleQuantity, bool &generatorOn);
        WorkerGen(const WorkerGen &other);
        ~WorkerGen();

        WorkerGen &operator=(const WorkerGen &other);

        void call(vec3 &gravityPos, bool gravityOn, float gravityStrength) const;
        void generate(int particlePerFrame);
        void init();

};