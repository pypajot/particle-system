#include <cuda_gl_interop.h>

#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>

#include "Worker/WorkerStatic.hpp"
#include "constants.hpp"

#define INIT_SIZE 1.0f

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

/// @brief Call the worker gravity and speed calculation
/// @param gravityArray The gravity points array
/// @note Maps and unmaps the cuda resources
void WorkerStatic::call(std::vector<Gravity> &gravityArray)
{
    Map();
    
    if (std::any_of(gravityArray.begin(), gravityArray.end(), checkGravityActive) && gravityArray.size() != 0)
    {
        cudaMemcpyToSymbol(gravity, gravityArray.data(), gravityArray.size() * sizeof(Gravity));
        GravityAction<<<dim3(gravityArray.size(), _blocks), _threadPerBlocks>>>(_buffer, _particleQty, _elemSize);
        checkCudaError("GravityAction kernel");
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