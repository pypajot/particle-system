#pragma once

#include <glm/vec3.hpp>

#define BASE_GRAVITY 0.6f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.2f
#define GRAVITY_STEP 0.2f

// Header only to prevent dynamic initilization of the constant array made of this class

/// @brief A gravity point
class Gravity
{
    public:
        /// @brief The position of the point in world coordinates
        glm::vec3 pos;
        /// @brief THe strength of the gravity
        float strength;
        /// @brief Is this point active
        bool active;
        
        Gravity() {}

        Gravity(const glm::vec3 &basePos) : pos(basePos)
        {
            strength = BASE_GRAVITY;
            active = true;
        }

        Gravity(const Gravity &other) : pos(other.pos)
        {
            strength = other.strength;
            active = other.active;
        }

        ~Gravity() {}

        Gravity &operator=(const Gravity &other)
        {
            if (this == &other)
                return *this;

            pos = other.pos;
            strength = other.strength;
            active = other.active;
            return *this;
        }

        /// @brief Set the position of the gravity point
        /// @param newPos The new position
        void SetPos(glm::vec3 newPos)
        {
            pos = newPos;
        }

        /// @brief Increment the strength of the gravity point
        void GravityUp()
        {
            if (strength == MAX_GRAVITY)
                return;
            strength += GRAVITY_STEP;
            if (strength >= MAX_GRAVITY)
                strength = MAX_GRAVITY;
        }

        /// @brief Decrement the strength of the gravity point
        void GravityDown()
        {
            if (strength == MIN_GRAVITY)
                return;
            strength -= GRAVITY_STEP;
            if (strength <= MIN_GRAVITY)
                strength = MIN_GRAVITY;
        }
};
