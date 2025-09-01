#pragma once

#include <vector_types.h>
#include <driver_types.h>

#include "vec3.hpp"

class GravityWorker
{
    public:
        bool managesBuffer;

        int threadPerBlocks;
        int blocks;
        int particleQty;
        cudaGraphicsResource *cudaGL_ptr;;

        GravityWorker();
        GravityWorker(GLuint VBO, int particleQuantity);
        GravityWorker(GravityWorker &other);
        ~GravityWorker();

        GravityWorker &operator=(GravityWorker &other);

        void call(vec3 &gravityPos);
};