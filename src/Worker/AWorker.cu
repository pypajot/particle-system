#include <iostream>

#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Worker/AWorker.cuh"

#define GRAVITY_RADIUS 0.25f

/// @brief The gavity points array in
__constant__ Gravity gravity[MAX_GRAVITY_POINTS];

/// @brief Get the last cuda error and display it
/// @param function The function after which the check was performed, used for information purposes
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

AWorker::AWorker(AWorker &&other) :
    _particleQty(other._particleQty),
    _elemSize(other._elemSize),
    _threadPerBlocks(other._threadPerBlocks),
    _blocks(other._blocks)
{
    other._managesBuffer = false;

    _cudaGL_ptr = other._cudaGL_ptr;
    _d_state = other._d_state;

    _managesBuffer = true;
}

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

    _particleQty = other._particleQty;
    _elemSize = other._elemSize;
    _threadPerBlocks = other._threadPerBlocks;
    _blocks = other._blocks;

    _cudaGL_ptr = other._cudaGL_ptr;
    _d_state = other._d_state;

    _managesBuffer = false;

    return *this;
}

AWorker &AWorker::operator=(AWorker &&other)
{
    if (this == &other)
        return *this;

    other._managesBuffer = false;

    _particleQty = other._particleQty;
    _elemSize = other._elemSize;
    _threadPerBlocks = other._threadPerBlocks;
    _blocks = other._blocks;

    _cudaGL_ptr = other._cudaGL_ptr;
    _d_state = other._d_state;

    _managesBuffer = true;

    return *this;
}

/// @brief Wrapper around cudaGraphicsMapResources and cudaGraphicsResourceGetMappedPointer
void AWorker::Map()
{
    size_t bufferSize = _particleQty * _elemSize * sizeof(float);
    
    cudaGraphicsMapResources(1, &_cudaGL_ptr);
    checkCudaError("Map resource");
    cudaGraphicsResourceGetMappedPointer((void **)&_buffer, &bufferSize, _cudaGL_ptr);
    checkCudaError("Get Mapped pointer");
}

/// @brief Wrapper around cudaGraphicsUnmapResources
void AWorker::Unmap()
{
    cudaGraphicsUnmapResources(1, &_cudaGL_ptr);
    checkCudaError("Unmap resource");
}

/// @brief Check the activity status of a gravity point
/// @param gravity Thje gravity point to check
/// @return True if active, false if not
bool checkGravityActive(const Gravity &gravityPoint)
{
    return gravityPoint.active;
}

/// @brief The gravity calulation kernel
/// @param buffer The buffer containing all particles
/// @param bufferIndexMax The maximum number of particle
/// @param elemSize The size of a particle in the buffer
__global__ void GravityAction(float *buffer, int bufferIndexMax, int elemSize)
{
    int index = blockIdx.y * blockDim.x + threadIdx.x;
    
    int gravityIndex = blockIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * elemSize;
    
    if (!gravity[gravityIndex].active)
        return;
    
    float distanceX = current[0] - gravity[gravityIndex].pos.x;
    float distanceY = current[1] - gravity[gravityIndex].pos.y;
    float distanceZ = current[2] - gravity[gravityIndex].pos.z;

    float d2 = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    float gravityForce = gravity[gravityIndex].strength / d2;
    
    // We use a spherical gravity instead of a punctual one to avoid divergence due to the discrete time step
    float distance = sqrt(d2);
    if (distance >= GRAVITY_RADIUS)
        gravityForce /= distance;    
    else
        gravityForce *=  powf(distance, 2) / powf(GRAVITY_RADIUS, 3);    
    
    gravityForce *=  TIME_FACTOR;
    current[3] -= distanceX * gravityForce;
    current[4] -= distanceY * gravityForce;
    current[5] -= distanceZ * gravityForce;
}