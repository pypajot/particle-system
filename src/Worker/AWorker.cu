#include <iostream>

#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Worker/AWorker.hpp"

#define TIME_FACTOR 1.0f / 60.0f

__device__
float uniformDisToBounds(float input, float min, float max)
{
    return input * (max - min) + min;
}

__global__
void InitRand(curandState *d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(0, index, 0, &d_state[index]);
}

void checkCudaError(const char *function)
{
    cudaError_t error = cudaGetLastError();

    if (error == cudaSuccess)
        return;

    const char *name = cudaGetErrorName(error);
    const char *string = cudaGetErrorString(error);
    std::cout << "In function " << function << "\nError " << name << " : " << string << "\n"; 
}

AWorker::AWorker()
{
    managesBuffer = false;
}

AWorker::AWorker(GLuint VBO, int particleQuantity)
{
    particleQty = particleQuantity;

    threadPerBlocks = 1024;
    blocks = particleQty / threadPerBlocks + 1;
    gravityStrength = BASE_GRAVITY;

    cudaGraphicsGLRegisterBuffer(&cudaGL_ptr, VBO, cudaGraphicsRegisterFlagsNone);
    checkCudaError("Register buffer");

    cudaMalloc(&d_state, sizeof(curandState) * threadPerBlocks * blocks);
    InitRand<<<blocks, threadPerBlocks>>>(d_state);

    managesBuffer = true;
}

AWorker::AWorker(const AWorker &other)
{
    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    cudaGL_ptr = other.cudaGL_ptr;
    d_state = other.d_state;
    gravityStrength = other.gravityStrength;

    managesBuffer = false;
}

AWorker::AWorker(AWorker &&other)
{
    other.managesBuffer = false;

    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    cudaGL_ptr = other.cudaGL_ptr;
    d_state = other.d_state;
    gravityStrength = other.gravityStrength;
        managesBuffer = true;
}

AWorker::~AWorker()
{
    if (managesBuffer)
    {
        cudaGraphicsUnregisterResource(cudaGL_ptr);
        checkCudaError("Unregister resource");
        cudaFree(d_state);
    }
}

AWorker &AWorker::operator=(const AWorker &other)
{
    if (this == &other)
        return *this;

    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    cudaGL_ptr = other.cudaGL_ptr;
    d_state = other.d_state;
    gravityStrength = other.gravityStrength;

    managesBuffer = false;

    return *this;
}

AWorker &AWorker::operator=(AWorker &&other)
{
    if (this == &other)
        return *this;

    other.managesBuffer = false;

    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    cudaGL_ptr = other.cudaGL_ptr;
    d_state = other.d_state;
    gravityStrength = other.gravityStrength;

    managesBuffer = true;

    return *this;
}

void GravityUp()
{
    if (gravityStrength >= MAX_GRAVITY)
        return;
    gracityStrength += 0.1f;
}

void GravityDown()
{
    if (gravityStrength <= MIN_GRAVITY)
        return;
    gracityStrength -= 0.1f;
}
