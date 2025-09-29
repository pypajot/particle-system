#include "Gravity.hpp"
// #pragma once
// #include "math/vec3.hpp"

// #define BASE_GRAVITY 1.0f
// #define MAX_GRAVITY 2.0f
// #define MIN_GRAVITY 0.3f
// #define GRAVITY_STEP 0.1f

/// @brief A gravity point
// class Gravity
// {
//     public:
//         /// @brief The position of the point in world coordinates
//         vec3 pos;
//         /// @brief THe strength of the gravity
//         float strength;
//         /// @brief Is this point active
//         bool active;
        
//         __device__ __host__ Gravity()
//         {
//             pos = vec3(0, 0, 0);
//             strength = BASE_GRAVITY;
//             active = false;
//         }

//         Gravity(const vec3 &basePos)
//         {
//             pos = basePos;
//             strength = BASE_GRAVITY;
//             active = true;
//         }

//         Gravity(const Gravity &other)
//         {
//             pos = other.pos;
//             strength = other.strength;
//             active = other.active;
//         }

//         ~Gravity() {}

//         Gravity &operator=(const Gravity &other)
//         {
//             if (this == &other)
//                 return *this;

//             pos = other.pos;
//             strength = other.strength;
//             active = other.active;
//             return *this;
//         }

//         /// @brief Set the position of the gravity point
//         /// @param newPos The new position
//         void SetPos(vec3 newPos)
//         {
//             pos = newPos;
//         }

//         /// @brief Increment the strength of the gravity point
//         void GravityUp()
//         {
//             if (strength == MAX_GRAVITY)
//                 return;
//             strength += GRAVITY_STEP;
//             if (strength >= MAX_GRAVITY)
//                 strength = MAX_GRAVITY;
//         }

//         /// @brief Decrement the strength of the gravity point
//         void GravityDown()
//         {
//             if (strength == MIN_GRAVITY)
//                 return;
//             strength -= GRAVITY_STEP;
//             if (strength <= MIN_GRAVITY)
//                 strength = MIN_GRAVITY;
//         }
// };

// bool checkActive(const Gravity &gravity);

Gravity::Gravity() : pos((glm::vec3(0, 0, 0))), strength(BASE_GRAVITY), active(false)
{
    // pos = glm::vec3(0, 0, 0);
    // strength = BASE_GRAVITY;
    // active = false;
}

Gravity::Gravity(const glm::vec3 &basePos)
{
    pos = basePos;
    strength = BASE_GRAVITY;
    active = true;
}

Gravity::Gravity(const Gravity &other)
{
    pos = other.pos;
    strength = other.strength;
    active = other.active;
}

Gravity::~Gravity()
{
}

Gravity &Gravity::operator=(const Gravity &other)
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
void Gravity::SetPos(const glm::vec3 &newPos)
{
    pos = newPos;
}

/// @brief Increment the strength of the gravity point
void Gravity::GravityUp()
{
    if (strength == MAX_GRAVITY)
        return;
    strength += GRAVITY_STEP;
    if (strength >= MAX_GRAVITY)
        strength = MAX_GRAVITY;
}

/// @brief Decrement the strength of the gravity point
void Gravity::GravityDown()
{
    if (strength == MIN_GRAVITY)
        return;
    strength -= GRAVITY_STEP;
    if (strength <= MIN_GRAVITY)
        strength = MIN_GRAVITY;
}

/// @brief Check the activity status of a gravity point
/// @param gravity Thje gravity point to check
/// @return True if active, false if not
bool checkActive(const Gravity &gravity)
{
    return gravity.active;
}