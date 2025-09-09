#pragma once

#include "math/vec3.hpp"

#define BASE_GRAVITY 1.0f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.3f

class Gravity
{
    public:
        vec3 pos;
        float strength;
        bool active;
        
        Gravity();
        Gravity(const vec3 &pos);
        Gravity(const Gravity &other);
        ~Gravity();

        Gravity &operator=(const Gravity &other);

        void SetPos(vec3 vewPos);
        
        void GravityUp();
        void GravityDown();
};

bool checkActive(const Gravity &gravity);