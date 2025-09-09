#pragma once

#include "Worker/AWorker.hpp"

class WorkerGen : public AWorker
{
    private:
        int _currentParticle;
        int _particlePerFrame;
        
    public:
        WorkerGen();
        WorkerGen(GLuint VBO, int particleQuantity);
        WorkerGen(const WorkerGen &other);
        WorkerGen(WorkerGen &&other);
        ~WorkerGen();

        WorkerGen &operator=(const WorkerGen &other);
        WorkerGen &operator=(WorkerGen &&other);

        void call(std::vector<Gravity> &gravity);
        void generate(int particlePerFrame);
        void init();

};