#pragma once

#include <driver_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "Gravity.cuh"
#include "constants.hpp"

#define THREAD_PER_BLOCK 1024

#define TIME_FACTOR 1.0f / TARGET_FRAMERATE

/// @brief Base class for a cuda worker that will be used to parallelize calculations
class AWorker
{
    protected:
        /// @brief The buffer management status
        bool _managesBuffer;
        /// @brief The OpenGL / Cuda interop resource
        cudaGraphicsResource *_cudaGL_ptr;
        /// @brief The buffer that will be used for cuda device functions
        float *_buffer;
        
        /// @brief The size of the elements of the buffer
        int _elemSize;
        /// @brief The number of particles, ie. the number of elements in the buffer
        int _particleQty;
        /// @brief The number of threads per blocks used for the different kernels
        int _threadPerBlocks;
        /// @brief The number of blocks used for he different kernels
        int _blocks;
        
        /// @brief The random state used for initialization
        curandState *_d_state;
        
    public:
        AWorker();
        AWorker(GLuint VBO, int particleQuantity, int elemSz);
        AWorker(const AWorker &other);
        AWorker(AWorker &&other);

        virtual ~AWorker();

        void Unregister();

        AWorker &operator=(const AWorker &other);
        AWorker &operator=(AWorker &&other);

        void Map();
        void Unmap();

        virtual void call(std::vector<Gravity> &gravity) = 0;
        virtual void init() = 0;

};

void checkCudaError(const char *function);