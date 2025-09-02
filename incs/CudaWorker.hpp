#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

#include "vec3.hpp"

class CudaWorker
{
    public:
        bool managesBuffer;

        int threadPerBlocks;
        int blocks;
        int particleQty;
        cudaGraphicsResource *cudaGL_ptr;
        curandState *d_state;
        int currentParticle;
        int particlePerFrame;

        CudaWorker();
        CudaWorker(GLuint VBO, int particleQuantity);
        CudaWorker(CudaWorker &other);
        ~CudaWorker();

        CudaWorker &operator=(CudaWorker &other);

        void call(vec3 &gravityPos, bool gravityOn);
        void init();
        void callGen(vec3 &gravityPos, bool gravityOn);
        void initGen(float maxTtl);

};