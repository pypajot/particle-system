#include <iostream>

#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Worker/AWorker.hpp"

#define TIME_FACTOR 1.0f / 60.0f

void checkCudaError(const char *function)
{
    cudaError_t error = cudaGetLastError();

    if (error == cudaSuccess)
        return;

    const char *name = cudaGetErrorName(error);
    const char *string = cudaGetErrorString(error);
    std::cerr << "In function " << function << "\nError " << name << " : " << string << "\n"; 
}

__global__
void InitRand(curandState *_d_state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(0, index, 0, &_d_state[index]);
}

__global__ 
void GravityAction(float *buffer, int bufferIndexMax, Gravity *gravity, int stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int gravityIndex = blockIdx.y;
    
    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * stride;

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

AWorker::AWorker() :
    _particleQty(),
    _elemSize(),
    _threadPerBlocks(),
    _blocks()
{
    _managesBuffer = false;
}

AWorker::AWorker(GLuint VBO, int particleQty, int elemSize) :
    _particleQty(particleQty),
    _threadPerBlocks(THREAD_PER_BLOCK),
    _blocks(_particleQty / _threadPerBlocks + 1),
    _elemSize(elemSize)
{
    cudaGraphicsGLRegisterBuffer(&_cudaGL_ptr, VBO, cudaGraphicsRegisterFlagsNone);
    checkCudaError("Register buffer");

    cudaMalloc(&_d_state, sizeof(curandState) * _threadPerBlocks * _blocks);
    InitRand<<<_blocks, _threadPerBlocks>>>(_d_state);

    _managesBuffer = true;
}

AWorker::AWorker(const AWorker &other) :
    _particleQty(other._particleQty),
    _elemSize(other._elemSize),
    _threadPerBlocks(other._threadPerBlocks),
    _blocks(other._blocks)
{
    _cudaGL_ptr = other._cudaGL_ptr;
    _d_state = other._d_state;

    _managesBuffer = false;
}

// AWorker::AWorker(AWorker &&other) :
//     _particleQty(other._particleQty),
//     _elemSize(other._elemSize),
//     _threadPerBlocks(other._threadPerBlocks),
//     _blocks(other._blocks)
// {
//     other._managesBuffer = false;

//     _cudaGL_ptr = other._cudaGL_ptr;
//     _d_state = other._d_state;

//     _managesBuffer = true;
// }

AWorker::~AWorker()
{
    if (_managesBuffer)
    {
        cudaGraphicsUnregisterResource(_cudaGL_ptr);
        checkCudaError("Unregister resource");
        cudaFree(_d_state);
    }
}

AWorker &AWorker::operator=(const AWorker &other)
{
    if (this == &other)
        return *this;

    _cudaGL_ptr = other._cudaGL_ptr;
    _d_state = other._d_state;

    _managesBuffer = false;

    return *this;
}

// AWorker &AWorker::operator=(AWorker &&other)
// {
//     if (this == &other)
//         return *this;

//     other._managesBuffer = false;

//     _cudaGL_ptr = other._cudaGL_ptr;
//     _d_state = other._d_state;

//     _managesBuffer = true;

//     return *this;
// }

void AWorker::Map()
{
    size_t bufferSize = _particleQty * _elemSize * sizeof(float);
    
    cudaGraphicsMapResources(1, &_cudaGL_ptr);
    checkCudaError("Map resource");
    cudaGraphicsResourceGetMappedPointer((void **)&_buffer, &bufferSize, _cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
}

void AWorker::Unmap()
{
    cudaGraphicsUnmapResources(1, &_cudaGL_ptr);
    checkCudaError("Unmap resource");
}

