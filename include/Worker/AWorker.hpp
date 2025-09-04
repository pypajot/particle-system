#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3;

class AWorker
{
    protected:
        bool managesBuffer;
        cudaGraphicsResource *cudaGL_ptr;

        int particleQty;
        int threadPerBlocks;
        int blocks;
        
        curandState *d_state;
        
    public:
        AWorker();
        AWorker(GLuint VBO, int particleQuantity);
        AWorker(const AWorker &other);
        AWorker(AWorker &&other);

        ~AWorker();

        AWorker &operator=(const AWorker &other);
        AWorker &operator=(AWorker &&other);

        virtual void call(vec3 &gravityPos, bool gravityOn, float gravityStrength) = 0;
        virtual void init() = 0;

};

void checkCudaError(const char *function);