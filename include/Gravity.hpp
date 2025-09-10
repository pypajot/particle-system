#pragma once

#include "math/vec3.hpp"

#define BASE_GRAVITY 1.0f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.3f
#define GRAVITY_STEP 0.1f

/// @brief A gravity point
class Gravity
{
    public:
        /// @brief The position of the point in world coordinates
        vec3 pos;
        /// @brief THe strength of the gravity
        float strength;
        /// @brief Is this point active
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