#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3;

class AWorker
{
    private:
        bool managesBuffer;
        cudaGraphicsResource *cudaGL_ptr;

        int particleQty;
        int threadPerBlocks;
        int blocks;

        curandState *d_state;
        
    public:
        Worker();
        Worker(GLuint VBO, int particleQuantity);
        Worker(const Worker &other);
        Worker(Worker &&other);

        ~Worker();

        Worker &operator=(const Worker &other);
        Worker &operator=(Worker &&other);

        virtual void call(vec3 &gravityPos, bool gravityOn) = 0;
        virtual void init() = 0;
        // void callGen(vec3 &gravityPos, bool gravityOn);
        // void initGen(float maxTtl);

};