#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>

#include "Worker/WorkerGen.hpp"

#define GRAVITY_FACTOR 1
#define TIME_FACTOR 1.0f / 60.0f

__device__
float uniformDisToBounds(float input, float min, float max)
{
    return input * (max - min) + min;
}

WorkerGen::WorkerGen() : AWorker()
{
}

WorkerGen::WorkerGen(GLuint VBO, int particleQuantity) : AWorker(VBO, particleQuantity, 7)
{
    _currentParticle = 0;
}

WorkerGen::WorkerGen(const WorkerGen &other) : AWorker(other)
{
    _currentParticle = other._currentParticle;
}

WorkerGen::WorkerGen(WorkerGen &&other) : AWorker(std::move(other))
{
    _currentParticle = other._currentParticle;
}

WorkerGen::~WorkerGen()
{
}

WorkerGen &WorkerGen::operator=(const WorkerGen &other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(other);
    _currentParticle = other._currentParticle;
    return *this;
}

WorkerGen &WorkerGen::operator=(WorkerGen &&other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(std::move(other));
    _currentParticle = other._currentParticle;
    return *this;
}

__device__
bool ParticleIsGenerated(int index, int currentParticle, int particlePerFrame, int bufferIndexMax)
{
    if (currentParticle + particlePerFrame < bufferIndexMax)
        return index >= currentParticle && index < currentParticle + particlePerFrame;
    else
        return index >= currentParticle || index < (currentParticle + particlePerFrame) % bufferIndexMax;
}

__global__ 
void GeneratorAction(float *buffer, int bufferIndexMax, curandState *d_state, int particlePerFrame, int currentParticle)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    
    if (ParticleIsGenerated(index, currentParticle, particlePerFrame, bufferIndexMax))
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
}

__global__
void LoopActionGen(float *buffer, int bufferIndexMax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;

    current[6] += 1;
}

__global__ 
void GravityActionGen(float *buffer, int bufferIndexMax, Gravity *gravity)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gravityIndex = blockIdx.y;
    
    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    if (gravity[gravityIndex].active)
    {
        float distanceX = current[0] - gravity[gravityIndex].pos.x;
        float distanceY = current[1] - gravity[gravityIndex].pos.y;
        float distanceZ = current[2] - gravity[gravityIndex].pos.z;
    
        float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    
        float speedFactor = TIME_FACTOR * gravity[gravityIndex].strength / distance;
    
        current[3] -= distanceX * speedFactor;
        current[4] -= distanceY * speedFactor;
        current[5] -= distanceZ * speedFactor;
    }
}

void WorkerGen::call(std::vector<Gravity> &gravity)
{
    Gravity *test;
    
    if (std::any_of(gravity.begin(), gravity.end(), checkActive) && gravity.size() != 0)
    {
        cudaMalloc(&test, gravity.size() * sizeof(Gravity));
        cudaMemcpy(test, gravity.data(), gravity.size() * sizeof(Gravity), cudaMemcpyHostToDevice);
        GravityActionGen<<<dim3(_blocks, gravity.size()), _threadPerBlocks>>>(_buffer, _particleQty, test);
        cudaFree(test);
    }
    LoopActionGen<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty);
}

void WorkerGen::generate(int particlePerFrame)
{
    GeneratorAction<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty, _d_state, particlePerFrame, _currentParticle);
    _currentParticle = (_currentParticle + particlePerFrame) % _particleQty;
}

__global__
void InitGenerator(float *buffer, int bufferIndexMax)
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
    current[6] = 301.0f;
}
#include <iostream>
void WorkerGen::init()
{
    Map();
    std::cout << _blocks << " " << _threadPerBlocks << "\n";
    InitGenerator<<<_blocks, _threadPerBlocks>>>(_buffer, _particleQty);
    checkCudaError("kernel");
    Unmap();
}
