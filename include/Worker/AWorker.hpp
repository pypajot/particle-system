#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3;

#define BASE_GRAVITY 1.0f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.3f

class AWorker
{
    protected:
        bool managesBuffer;
        cudaGraphicsResource *cudaGL_ptr;

        int particleQty;
        int threadPerBlocks;
        int blocks;
        

        curandState *d_state;
        
        float gravityStrength;

    public:
        AWorker();
        AWorker(GLuint VBO, int particleQuantity);
        AWorker(const AWorker &other);
        AWorker(AWorker &&other);

        ~AWorker();

        AWorker &operator=(const AWorker &other);
        AWorker &operator=(AWorker &&other);

        virtual void call(vec3 &gravityPos, bool gravityOn) = 0;
        virtual void init() = 0;

        void GravityUp();
        void GravityDown();
        // void callGen(vec3 &gravityPos, bool gravityOn);
        // void initGen(float maxTtl);

};