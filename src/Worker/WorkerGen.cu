#include <cuda_gl_interop.h>


#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#include "Worker/WorkerGen.hpp"

#define GRAVITY_FACTOR 1
#define TIME_FACTOR 1.0f / 60.0f

__device__
float uniformDisToBounds(float input, float min, float max)
{
    return input * (max - min) + min;
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

WorkerGen::WorkerGen() : AWorker()
{
}

WorkerGen::WorkerGen(GLuint VBO, int particleQuantity) : AWorker(VBO, particleQuantity)
{
    maxTtl = BASE_TTL;
    currentParticle = 0;
    particlePerFrame = BASE_PPF;
}

WorkerGen::WorkerGen(const WorkerGen &other) : AWorker(other)
{
    maxTtl = other.maxTtl;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
}

WorkerGen::WorkerGen(WorkerGen &&other) : AWorker(other)
{
    maxTtl = other.maxTtl;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
}

WorkerGen::~WorkerGen()
{
    if (managesBuffer)
    {
        cudaGraphicsUnregisterResource(cudaGL_ptr);
        checkCudaError("Unregister resource");
        cudaFree(d_state);
    }
}

WorkerGen &WorkerGen::operator=(const WorkerGen &other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(other);
    maxTtl = other.maxTtl;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    return *this;
}

WorkerGen &WorkerGen::operator=(WorkerGen &&other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(other);
    maxTtl = other.maxTtl;
    currentParticle = other.currentParticle;
    particlePerFrame = other.particlePerFrame;
    return *this;
}

__global__ 
void LoopActionGenerator(float *buffer, vec3 gravityPos, float gravityStrength, int bufferIndexMax, bool gravityOn, curandState *d_state, int particlePerFrame, int currentParticle)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    
    if (index >= currentParticle && index < currentParticle + particlePerFrame % bufferIndexMax)
    {
        float angleY = M_PI_2 - uniformDisToBounds(curand_uniform(&d_state[index]), 0, 0.2f);
        float angleXZ = uniformDisToBounds(curand_uniform(&d_state[index]), 0, M_PI * 2);
        float speed = uniformDisToBounds(curand_uniform(&d_state[index]), 0.5f, 1.0f);

        current[0] = 0.0f;
        current[1] = -0.5f;
        current[2] = 0.0f;
        current[3] = cos(angleY) * cos(angleXZ) * speed;
        current[4] = sin(angleY) * speed;
        current[5] = cos(angleY) * sin(angleXZ) * speed;
        current[6] = 0.0f;

    }

    if (gravityOn)
    {
        float distanceX = current[0] - gravityPos.x;
        float distanceY = current[1] - gravityPos.y;
        float distanceZ = current[2] - gravityPos.z;
    
        float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    
        float speedFactor = TIME_FACTOR * GRAVITY_FACTOR / distance;
    
        current[3] -= distanceX * speedFactor;
        current[4] -= distanceY * speedFactor;
        current[5] -= distanceZ * speedFactor;
    }

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;

    current[6] += 1;
}

void WorkerGen::call(vec3 &gravityPos, bool gravityOn)
{
    size_t bufferSize = particleQty * 7 * sizeof(float);
    float *buffer;
    
    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    LoopActionGenerator<<<blocks, threadPerBlocks>>>(buffer, gravityPos, gravityStrength, particleQty, gravityOn, d_state, particlePerFrame, currentParticle);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
    currentParticle = (currentParticle + particlePerFrame) % particleQty;
}

__global__
void InitGenerator(float *buffer, int bufferIndexMax, float maxTtl)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    current[0] = 0.0f;
    current[1] = 0.0f;
    current[2] = 0.0f;
    current[3] = 0.0f;
    current[4] = 0.0f;
    current[5] = 0.0f;
    current[6] = maxTtl;
}

void WorkerGen::init()
{
    size_t bufferSize = particleQty * 7 * sizeof(float);
    float *buffer;

    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");

    InitGen<<<blocks, threadPerBlocks>>>(buffer, particleQty, maxTtl);

    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
}
