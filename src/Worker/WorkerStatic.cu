#include <cuda_gl_interop.h>


#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#include "Worker/WorkerStatic.hpp"

#define GRAVITY_FACTOR 1
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

WorkerStatic::WorkerStatic() : AWorker()
{
}

WorkerStatic::WorkerStatic(GLuint VBO, int particleQuantity) : AWorker(VBO, particleQuantity)
{
}

WorkerStatic::WorkerStatic(WorkerStatic &other) : AWorker(other)
{
}

WorkerStatic::~WorkerStatic()
{
    if (managesBuffer)
    {
        cudaGraphicsUnregisterResource(cudaGL_ptr);
        checkCudaError("Unregister resource");
        cudaFree(d_state);
    }
}

WorkerStatic &WorkerStatic::operator=(WorkerStatic &other)
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

    managesBuffer = true;
    return *this;
}

// __global__
// void UpdatePosAndSpeed(float* current, float speedFactor)
// {
//     int index = threadIdx.x;
//     current[index + 3 = ]
// }

__global__ 
void GravityAction(float *buffer, vec3 gravityPos, int bufferIndexMax, bool gravityOn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;
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
}

void WorkerStatic::call(vec3 &gravityPos, bool gravityOn)
{
    size_t bufferSize = particleQty * 6 * sizeof(float);
    float *buffer;
    
    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    GravityAction<<<blocks, threadPerBlocks>>>(buffer, gravityPos, particleQty, gravityOn);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
}

__global__ 
void GravityActionGen(float *buffer, vec3 gravityPos, int bufferIndexMax, bool gravityOn, curandState *d_state, int particlePerFrame, int currentParticle)
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
        // std::uniform_real_distribution<> disY(0, 0.2f);
        // std::uniform_real_distribution<> disXZ(0, M_PI * 2);
        // std::uniform_real_distribution<> speedDis(0.5f, 1.0f);
        
        // index = (currentParticle + i) % particleQty;
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

void WorkerStatic::callGen(vec3 &gravityPos, bool gravityOn)
{
    size_t bufferSize = particleQty * 7 * sizeof(float);
    float *buffer;
    // (void)gravityPos;
    // curandState *d_state;
    // cudaMalloc(&d_state, sizeof(curandState) * threadPerBlocks * blocks);
    // InitRand<<<blocks, threadPerBlocks>>>(d_state);
    
    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    GravityActionGen<<<blocks, threadPerBlocks>>>(buffer, gravityPos, particleQty, gravityOn, d_state, particlePerFrame, currentParticle);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
    currentParticle = (currentParticle + particlePerFrame) % particleQty;
    // cudaFree(d_state);
}


__global__
void Init(float *buffer, int bufferIndexMax, curandState *d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    float angleY;
    float angleXZ;

    angleY = uniformDisToBounds(curand_uniform(&d_state[index]), -M_PI_2, M_PI_2);
    angleXZ = uniformDisToBounds(curand_uniform(&d_state[index]), 0, M_PI * 2);

    current[0] = cos(angleY) * cos(angleXZ);
    current[1] = sin(angleY);
    current[2] = cos(angleY) * sin(angleXZ);
    current[3] = uniformDisToBounds(curand_uniform(&d_state[index]), 0, 0.1f);
    current[4] = uniformDisToBounds(curand_uniform(&d_state[index]), 0, 0.1f);
    current[5] = uniformDisToBounds(curand_uniform(&d_state[index]), 0, 0.1f);
}

void WorkerStatic::init()
{
    size_t bufferSize = particleQty * 6 * sizeof(float);
    float *buffer;

    // curandState *d_state;
    // cudaMalloc(&d_state, sizeof(curandState) * threadPerBlocks * blocks);
    // InitRand<<<blocks, threadPerBlocks>>>(d_state);

    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    Init<<<blocks, threadPerBlocks>>>(buffer, particleQty, d_state);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
}

__global__
void InitGen(float *buffer, int bufferIndexMax, float maxTtl)
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

void WorkerStatic::initGen(float maxTtl)
{
    size_t bufferSize = particleQty * 7 * sizeof(float);
    float *buffer;
    // curandState *d_state;
    // cudaMalloc(&d_state, sizeof(curandState) * threadPerBlocks * blocks);
    // InitRand<<<blocks, threadPerBlocks>>>(d_state);

    cudaGraphicsMapResources(1, &cudaGL_ptr);
    checkCudaError("Map resource");

    cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
    InitGen<<<blocks, threadPerBlocks>>>(buffer, particleQty, maxTtl);
    cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    checkCudaError("Unmap resource");
}
