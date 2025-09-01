#include <cuda_gl_interop.h>

#include "GravityWorker.hpp"

#include <iostream>

#define GRAVITY_FACTOR 1
#define TIME_FACTOR 1.0f / 60.0f

void checkCudaError(const char *function)
{
    cudaError_t error = cudaGetLastError();

    if (error == cudaSuccess)
        return;

    const char *name = cudaGetErrorName(error);
    const char *string = cudaGetErrorString(error);
    std::cout << "In function " << function << "\nError " << name << " : " << string << "\n"; 
}

GravityWorker::GravityWorker()
{
    managesBuffer = false;
}

GravityWorker::GravityWorker(GLuint VBO, int particleQuantity)
{
    particleQty = particleQuantity;

    threadPerBlocks = 256;
    blocks = particleQty / threadPerBlocks + 1;
    cudaGraphicsGLRegisterBuffer(&cudaGL_ptr, VBO, cudaGraphicsRegisterFlagsNone);
    checkCudaError("Register buffer");
    managesBuffer = true;
}

GravityWorker::GravityWorker(GravityWorker &other)
{
    other.managesBuffer = false;

    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    cudaGL_ptr = other.cudaGL_ptr;
    managesBuffer = true;
}

GravityWorker::~GravityWorker()
{
    if (managesBuffer)
    {
        cudaGraphicsUnregisterResource(cudaGL_ptr);
        checkCudaError("Unregister resource");
    }
}

GravityWorker &GravityWorker::operator=(GravityWorker &other)
{
    if (this == &other)
        return *this;

    other.managesBuffer = false;

    particleQty = other.particleQty;
    threadPerBlocks = other.threadPerBlocks;
    blocks = other.blocks;
    cudaGL_ptr = other.cudaGL_ptr;
    managesBuffer = true;
    return *this;
}

__global__ 
void GravityAction(float *buffer, vec3 gravityPos, int bufferIndexMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;
    float distanceX = current[0] - gravityPos.x;
    float distanceY = current[1] - gravityPos.y;
    float distanceZ = current[2] - gravityPos.z;

    float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);

    float speedFactor = TIME_FACTOR * GRAVITY_FACTOR / distance;

    current[3] -= distanceX * speedFactor;
    current[4] -= distanceY * speedFactor;
    current[5] -= distanceZ * speedFactor;

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;
}

void GravityWorker::call(vec3 &gravityPos)
{
    size_t bufferSize = particleQty * 6 * sizeof(float);
    float *buffer;
    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");
    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    GravityAction<<<blocks, threadPerBlocks>>>(buffer, gravityPos, particleQty);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
}
