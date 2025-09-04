#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3;

#define THREAD_PER_BLOCK 1024

class AWorker
{
    protected:
        bool managesBuffer;
        cudaGraphicsResource *cudaGL_ptr;
        float *buffer;
        
        const int elemSize;
        const int particleQty;
        const int threadPerBlocks;
        const int blocks;
        
        curandState *d_state;
        
    public:
        AWorker();
        AWorker(GLuint VBO, int particleQuantity, int elemSz);
        AWorker(const AWorker &other);
        AWorker(AWorker &&other);

        ~AWorker();

        AWorker &operator=(const AWorker &other);
        AWorker &operator=(AWorker &&other);

        void Map();
        void Unmap();

        virtual void call(vec3 &gravityPos, bool gravityOn, float gravityStrength) = 0;
        virtual void init() = 0;

};
