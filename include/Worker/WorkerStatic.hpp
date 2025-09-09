#pragma once

#include "Worker/AWorker.hpp"

class WorkerStatic : public AWorker
{
    public:

        WorkerStatic();
        WorkerStatic(GLuint VBO, int particleQuantity);
        WorkerStatic(const WorkerStatic &other);
        WorkerStatic(WorkerStatic &&other);
        ~WorkerStatic();

        WorkerStatic &operator=(const WorkerStatic &other);
        WorkerStatic &operator=(WorkerStatic &&other);

        void call(std::vector<Gravity> &gravity);
        void init();
        void initCube();

};