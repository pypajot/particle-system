#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3;

#define THREAD_PER_BLOCK 1024

class AWorker
{
    protected:
        bool _managesBuffer;
        cudaGraphicsResource *_cudaGL_ptr;
        float *_buffer;
        
        const int _elemSize;
        const int _particleQty;
        const int _threadPerBlocks;
        const int _blocks;
        
        curandState *_d_state;
        
    public:
        AWorker();
        AWorker(GLuint VBO, int particleQuantity, int elemSz);
        AWorker(const AWorker &other);
        AWorker(AWorker &&other);

        virtual ~AWorker();

        AWorker &operator=(const AWorker &other);
        AWorker &operator=(AWorker &&other);

        void Map();
        void Unmap();

        virtual void call(std::vector<Gravity> &gravity) = 0;
        virtual void init() = 0;

};
