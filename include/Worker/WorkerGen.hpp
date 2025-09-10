#pragma once

#include "Worker/AWorker.hpp"

/// @brief The class used to parallelize calculation par a system with a generator
class WorkerGen : public AWorker
{
    private:
        /// @brief The index used to track the next p[rticle to be generated
        int _currentParticle;
        
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