#include <cuda_gl_interop.h>

#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>

#include "Worker/WorkerStatic.hpp"

#define TIME_FACTOR 1.0f / 60.0f
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

// WorkerStatic::WorkerStatic(WorkerStatic &&other) : AWorker(other)
// {
// }

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

// WorkerStatic &WorkerStatic::operator=(WorkerStatic &&other)
// {
//     if (this == &other)
//         return *this;

//     AWorker::operator=(other);
//     return *this;
// }

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

void WorkerStatic::call(std::vector<Gravity> &gravity)
{
    Map();
    if (std::any_of(gravity.begin(), gravity.end(), checkActive) && gravity.size() != 0)
        GravityAction<<<dim3(_blocks, gravity.size()), _threadPerBlocks>>>(_buffer, _particleQty, gravity.data(), _elemSize);
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

    int side = uniformDisToBoundsI(curand_uniform(&d_state[index]), 1, 6);

    if (side == 1)
    {
        current[0] = -INIT_SIZE;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, INIT_SIZE); 
    }
    else if (side == 2)
    {
        current[0] = INIT_SIZE;
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE); 
    }
    else if (side == 3)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[1] = -INIT_SIZE;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE); 
    }
    else if (side == 4)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[1] = INIT_SIZE;
        current[2] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE); 
    }
    else if (side == 5)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[2] = -INIT_SIZE; 
    }
    else if (side == 6)
    {
        current[0] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[1] = uniformDisToBoundsF(curand_uniform(&d_state[index]), -INIT_SIZE, INIT_SIZE);
        current[2] = INIT_SIZE; 
    }
    current[3] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[4] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
    current[5] = uniformDisToBoundsF(curand_uniform(&d_state[index]), 0, 0.1f);
}

void WorkerStatic::initCube()
{
    Map();
    InitCube<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty, _d_state);
    Unmap();
}