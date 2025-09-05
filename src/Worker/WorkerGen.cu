#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

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

WorkerGen::WorkerGen(GLuint VBO, int particleQuantity, float maxTtl, int particlePerFrame, bool &generatorOn) : AWorker(VBO, particleQuantity, 7)
{
    currentParticle = 0;
}

WorkerGen::WorkerGen(const WorkerGen &other) : AWorker(other)
{
    currentParticle = other.currentParticle;
}

WorkerGen::WorkerGen(WorkerGen &&other) : AWorker(other)
{
    currentParticle = other.currentParticle;
}

WorkerGen::~WorkerGen()
{
    if (_managesBuffer)
    {
        cudaGraphicsUnregisterResource(_cudaGL_ptr);
        checkCudaError("Unregister resource");
        cudaFree(d_state);
    }
}

WorkerGen &WorkerGen::operator=(const WorkerGen &other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(other);
    currentParticle = other.currentParticle;
    return *this;
}

WorkerGen &WorkerGen::operator=(WorkerGen &&other)
{
    if (this == &other)
        return *this;

    this->AWorker::operator=(other);
    currentParticle = other.currentParticle;
    return *this;
}

__device__
bool ParticleIsGenerated(int index, int currentParticle, int particlePerframe, int bufferIndexMax)
{
    if (currentParticle + particlePerFrame < bufferIndexMax)
        return index >= currentParticle && index < currentParticle + particlePerFrame;
    else
        return index >= currentParticle || index < currentParticle + particlePerFrame % bufferIndexMax;
}

__global__ 
void LoopActionGenerator(float *buffer, int bufferIndexMax, curandState *d_state, int particlePerFrame, int currentParticle)
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
void LoopActionGravity(float *buffer, vec3 gravityPos, float gravityStrength, int bufferIndexMax, bool gravityOn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= bufferIndexMax)
        return;

    float *current = buffer + index * 7;

    if (gravityOn)
    {
        float distanceX = current[0] - gravityPos.x;
        float distanceY = current[1] - gravityPos.y;
        float distanceZ = current[2] - gravityPos.z;
    
        float distance = powf(distanceX, 2) + powf(distanceY, 2) + powf(distanceZ, 2);
    
        float speedFactor = TIME_FACTOR * gravityStrength / distance;
    
        current[3] -= distanceX * speedFactor;
        current[4] -= distanceY * speedFactor;
        current[5] -= distanceZ * speedFactor;
    }

    current[0] += current[3] * TIME_FACTOR;
    current[1] += current[4] * TIME_FACTOR;
    current[2] += current[5] * TIME_FACTOR;

    current[6] += 1;
}


void WorkerGen::call(vec3 &gravityPos, bool gravityOn, float gravityStrength) const
{
    // size_t bufferSize = particleQty * 7 * sizeof(float);
    // float *buffer;
    
    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");
    LoopActionGravity<<<blocks, threadPerBlocks>>>(buffer, gravityPos, gravityStrength, particleQty, gravityOn);
    // cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    // checkCudaError("Unmap resource");
    // if (generatorOn)
    //     currentParticle = (currentParticle + particlePerFrame) % particleQty;
}

void WorkerGen::generate(int particlePerFrame)
{
    // size_t bufferSize = particleQty * 7 * sizeof(float);
    // float *buffer;
    
    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");
    LoopActionGenerator<<<blocks, threadPerBlocks>>>(buffer, particleQty, d_state, particlePerFrame, currentParticle);
    // cudaGraphicsUnmapResources(1, &cudaGL_ptr);
    // checkCudaError("Unmap resource");
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
    current[6] = __INT32_MAX__;
}

void WorkerGen::init()
{
    // size_t bufferSize = particleQty * 7 * sizeof(float);
    // float *buffer;
    AWorker::Map();
    // cudaGraphicsMapResources(1, &cudaGL_ptr);
    // checkCudaError("Map resource");

    // cudaGraphicsResourceGetMappedPointer((void **)&buffer, &bufferSize, cudaGL_ptr);
    // checkCudaError("Get Mapped pointer");

    InitGen<<<blocks, threadPerBlocks>>>(buffer, particleQty, maxTtl);
    AWorker::Unmap();
}
