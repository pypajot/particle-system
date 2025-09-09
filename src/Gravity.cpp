#include "Gravity.hpp"

Gravity::Gravity()
{
    pos = vec3(0, 0, 0);
    strength = BASE_GRAVITY;
    active = false;
}

Gravity::Gravity(const vec3 &basePos)
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


void Gravity::SetPos(vec3 newPos)
{
    pos = newPos;
}


void Gravity::GravityUp()
{
    if (strength >= MAX_GRAVITY)
        return;
    strength += 0.1f;
}

void Gravity::GravityDown()
{
    if (strength <= MIN_GRAVITY)
        return;
    strength -= 0.1f;
}

bool checkActive(const Gravity &gravity)
{
    return gravity.active;
}