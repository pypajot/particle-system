#include <cuda_gl_interop.h>

#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>

#include "Worker/WorkerStatic.hpp"
#include "constants.hpp"

#define INIT_SIZE 1.0f

__constant__ Gravity gravity[MAX_GRAVITY_POINTS];

__device__
float uniformDisToBoundsF(float input, float min, float max)
{
    return input * (max - min) + min;
}

__device__
float uniformDisToBoundsI(float input, int min, int max)
{
    float resultf = input * (max - min + 1.0f) + min;
    return floorf(resultf);
}

WorkerStatic::WorkerStatic() : AWorker()
{
}

WorkerStatic::WorkerStatic(GLuint VBO, int particleQuantity) : AWorker(VBO, particleQuantity, 6)
{
}

WorkerStatic::WorkerStatic(const WorkerStatic &other) : AWorker(other)
{
    // _particleQty = other._particleQty;
    // _elemSize = other._elemSize;
    // _threadPerBlocks = other._threadPerBlocks;
    // _blocks = other._blocks;
}

WorkerStatic::WorkerStatic(WorkerStatic &&other) : AWorker(std::move(other))
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

    AWorker::operator=(std::move(other));
    return *this;
}

__global__
void LoopActionStatic(float *buffer, int bufferIndexMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;
}

__global__ 
void GravityActionStatic(float *buffer, int bufferIndexMax)
{
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    int gravityIndex = blockIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;

    if (gravity[gravityIndex].active)
    {
        float distanceX = current[0] - gravity[gravityIndex].pos.x;
        float distanceY = current[1] - gravity[gravityIndex].pos.y;
        float distanceZ = current[2] - gravity[gravityIndex].pos.z;
    
        float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    
        float gravityForce = gravity[gravityIndex].strength / distance;
        gravityForce *= TIME_FACTOR;
    
        current[3] -= distanceX * gravityForce;
        current[4] -= distanceY * gravityForce;
        current[5] -= distanceZ * gravityForce;
    }
}

/// @brief Call the worker gravity and speed calculation
/// @param gravity The gravity points array
/// @note Maps and unmaps the cuda resources
void WorkerStatic::call(std::vector<Gravity> &gravityArray)
{
    Map();
    
    if (std::any_of(gravityArray.begin(), gravityArray.end(), checkActive) && gravityArray.size() != 0)
    {
        // cudaMalloc(&test, gravity.size() * sizeof(Gravity));
        cudaMemcpyToSymbol(gravity, gravityArray.data(), gravityArray.size() * sizeof(Gravity));
        GravityActionStatic<<<dim3(gravityArray.size(), _blocks), _threadPerBlocks>>>(_buffer, _particleQty);
        // cudaFree(test);
    }
    LoopActionStatic<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty);
    Unmap();
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

/// @brief Initialize the simulation to a sphere
/// @note Maps and unmaps the cuda resources
void WorkerStatic::init()
{
    Map();
    InitSphere<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty, _d_state);
    Unmap();
}

__global__
void InitCube(float *buffer, int bufferIndexMax, curandState *d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 6;
    float size = INIT_SIZE * 0.7;
    int side = uniformDisToBoundsI(curand_uniform(&d_state[index]), 1, 6);

    if (side == 1)
    {
        current[0] = -size;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size); 
    }
    else if (side == 2)
    {
        current[0] = size;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size); 
    }
    else if (side == 3)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[1] = -size;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size); 
    }
    else if (side == 4)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[1] = size;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size); 
    }
    else if (side == 5)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[2] = -size; 
    }
    else if (side == 6)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -size, size);
        current[2] = size; 
    }
    current[3] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[4] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[5] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
}

/// @brief Initialize the simulation to a sphere
/// @note Maps and unmaps the cuda resources
void WorkerStatic::initCube()
{
    Map();
    InitCube<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty, _d_state);
    Unmap();
}