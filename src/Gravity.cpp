#include "Gravity.hpp"

Gravity::Gravity()
{
    _pos = vec3(0, 0, 0);
    _strength = BASE_GRAVITY;
    active = false;
}

Gravity::Gravity(vec3 pos)
{
    _pos = pos;
    _strength = BASE_GRAVITY;
    active = true;
}

Gravity::Gravity(const Gravity &other)
{
    _pos = other._pos;
    _strength = other._strength;
    active = other.active;
}

Gravity::~Gravity()
{
}


Gravity &Gravity::operator=(const Gravity &other)
{
    if (this == &other)
        return *this;

    _pos = other._pos;
    _strength = other._strength;
    active = other.active;
    return *this;
}


void Gravity::SetPos(vec3 newPos)
{
    _pos = newPos;
}


void Gravity::GravityUp()
{
    if (_strength >= MAX_GRAVITY)
        return;
    _strength += 0.1f;
}

void Gravity::GravityDown()
{
    if (_strength <= MIN_GRAVITY)
        return;
    _strength -= 0.1f;
}
