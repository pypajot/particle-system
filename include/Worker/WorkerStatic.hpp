#pragma once

#include "Worker/AWorker.hpp"

class vec3;

class WorkerStatic : AWorker
{
    public:

        WorkerStatic();
        WorkerStatic(GLuint VBO, int particleQuantity);
        WorkerStatic(const WorkerStatic &other);
        WorkerStatic(WorkerStatic &&other);
        ~WorkerStatic();

        WorkerStatic &operator=(const WorkerStatic &other);
        WorkerStatic &operator=(WorkerStatic &&other);

        void call(vec3 &gravityPos, bool gravityOn);
        void init();

};