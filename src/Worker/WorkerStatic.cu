#include <cuda_gl_interop.h>


#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#include "Worker/WorkerStatic.hpp"

#define TIME_FACTOR 1.0f / 60.0f
#define INIT_SIZE 1.0f

__device__
float uniformDisToBoundsF(float input, float min, float max)
{
    return input * (max - min) + min;
}

__device__
float uniformDisToIBounds(float input, int min, int max)
{
    float resultf = input * (max - min + 1.0f) + min;
    return floorf(resultf);
}

void checkCudaError(const char *function)
{
    cudaError_t error = cudaGetLastError();

    if (error == cudaSuccess)
        return;

    const char *name = cudaGetErrorName(error);
    const char *string = cudaGetErrorString(error);
    std::cerr << "In function " << function << "\nError " << name << " : " << string << "\n"; 
}

WorkerStatic::WorkerStatic() : AWorker()
{
}

WorkerStatic::WorkerStatic(GLuint VBO, int particleQuantity) : AWorker(VBO, particleQuantity, 6)
{
}

WorkerStatic::WorkerStatic(const WorkerStatic &other) : AWorker(other)
{
}

WorkerStatic::WorkerStatic(WorkerStatic &&other) : AWorker(other)
{
}

WorkerStatic::~WorkerStatic()
{
}

WorkerStatic &WorkerStatic::operator=(const WorkerStatic &other)
{
    if (this == &other)
        return *this;

    AWorker::operator=(other);
    return *this;
}

WorkerStatic &WorkerStatic::operator=(WorkerStatic &&other)
{
    if (this == &other)
        return *this;

    AWorker::operator=(other);
    return *this;
}

__global__ 
void GravityAction(float *buffer, int bufferIndexMax, std::vector<Gravity> gravity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gravityIndex = blockIdx.y;
    
    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    if (gravity[gravityIndex].active)
    {
        float distanceX = current[0] - gravity[gravityIndex]._pos.x;
        float distanceY = current[1] - gravity[gravityIndex]._pos.y;
        float distanceZ = current[2] - gravity[gravityIndex]._pos.z;
    
        float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    
        float speedFactor = TIME_FACTOR * gravityStrength / distance;
    
        current[3] -= distanceX * speedFactor;
        current[4] -= distanceY * speedFactor;
        current[5] -= distanceZ * speedFactor;
    }
}

__global__
void LoopAction(float *buffer, int bufferIndexMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;
}

void WorkerStatic::call(std::vector<Gravity> &gravity)
{
    // size_t bufferSize = particleQty * 6 * sizeof(float);
    // float *buffer;
    
    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");
    AWorker::Map();
    if (std::any_of(gravity.begin(), gravity.end(), checkActive) && gravity.length() != 0)
        GravityAction<<<dim2(blocks, gravity.length()), threadPerBlocks>>>(buffer, particleQty, gravity);
    LoopAction<<<blocks, threadPerBlocks>>>(buffer, particleQty);
    AWorker::Unmap();
    // cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    // checkCudaError("Unmap resource");
}

__global__
void InitSphere(float *buffer, int bufferIndexMax, curandState *d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    float angleY;
    float angleXZ;

    angleY = uniformDisToBoundsF(curand_uniform(&d_state[index]), -M_PI_2, M_PI_2);
    angleXZ = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, M_PI * 2);

    current[0] = cos(angleY) * cos(angleXZ) * INIT_SIZE;
    current[1] = sin(angleY) * INIT_SIZE;
    current[2] = cos(angleY) * sin(angleXZ) * INIT_SIZE;
    current[3] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[4] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[5] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
}

void WorkerStatic::init()
{
    // size_t bufferSize = particleQty * 6 * sizeof(float);
    // float *buffer;

    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");
    AWorker::Map();
    InitSphere<<<blocks, threadPerBlocks>>>(buffer, particleQty, _d_state);
    AWorker::Unmap();
    // cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    // checkCudaError("Unmap resource");
}

__global__
void InitCube(float *buffer, int bufferIndexMax, curandState *d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    int side = uniformDisToBoundsI(curand_uniform(&d_state[index]), 1, 6);

    if (side == 1)
    {
        current[0] = -size;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE); 
    }
    else if (side == 2)
    {
        current[0] = size;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE); 
    }
    else if (side == 3)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[1] = -size;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE); 
    }
    else if (side == 4)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[1] = size;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE); 
    }
    else if (side == 5)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[2] = -size; 
    }
    else if (side == 6)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[2] = size; 
    }
    current[3] = speeduniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[4] = speeduniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[5] = speeduniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
}

void WorkerStatic::initCube()
{
    // size_t bufferSize = particleQty * 6 * sizeof(float);
    // float *buffer;

    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");
    AWorker::Map();
    InitCube<<<blocks, threadPerBlocks>>>(buffer, particleQty, _d_state);
    AWorker::Unmap();
    // cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    // checkCudaError("Unmap resource");
}