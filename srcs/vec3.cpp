#include <cmath>

#include "vec3.hpp"

vec3::vec3()
{
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}
vec3::vec3(float val)
{
    x = val;
    y = val;
    z = val;
}
vec3::vec3(float x, float y, float z)
{
    x = x;
    y = y;
    z = z;
}

vec3::vec3(const vec4 &vector)
{
    x = vector.x;
    y = vector.y;
    z = vector.z;
}


vec3::vec3(const vec3 &other)
{
    x = other.x;
    y = other.y;
    z = other.z;
}

vec3::~vec3()
{
}


vec3 &vec3::operator=(const vec3 &other)
{
    if (&other == this)
        return *this;

    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

vec3 &vec3::operator+=(const vec3 &other)
{
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

vec3 operator+(vec3 lhs, vec3& rhs)
{
    lhs += rhs;
    return lhs;
}

float vec3::length()
{
    return sqrt(x * x + y * y + z * z);
}


float *vec3::getValue()
{
    value[0] = x;
    value[1] = y;
    value[2] = z;
    return value;
}


