#include <cmath>

#include "math/vec4.hpp"
#include "math/vec3.hpp"

constexpr vec3::vec3()
{    
}
// {
//     // x = 0.0f;
//     // y = 0.0f;
//     // z = 0.0f;
// }
vec3::vec3(float val)
{
    x = val;
    y = val;
    z = val;
}
vec3::vec3(float a, float b, float c)
{
    x = a;
    y = b;
    z = c;
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

vec3 operator+(vec3 lhs, const vec3 &rhs)
{
    lhs += rhs;
    return lhs;
}

vec3 &vec3::operator-=(const vec3 &other)
{
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

vec3 operator-(vec3 lhs, vec3 const &rhs)
{
    lhs -= rhs;
    return lhs;
}

vec3 &vec3::operator*=(float scalar)
{
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

vec3 operator*(vec3 lhs, float scalar)
{
    lhs *= scalar;
    return lhs;
}

vec3 operator*(float scalar, vec3 rhs)
{
    rhs *= scalar;
    return rhs;
}

float dot(vec3 lhs, vec3 rhs)
{
   return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

vec3 cross(vec3 lhs, vec3 rhs)
{
    vec3 crossProduct(lhs.y * rhs.z - lhs.z * rhs.y,
                        lhs.z * rhs.x - lhs.x * rhs.z,
                        lhs.x * rhs.y - lhs.y * rhs.x);
    
    return crossProduct;
}


float vec3::length() const
{
    return sqrt(x * x + y * y + z * z);
}

vec3 vec3::normalize()
{
    vec3 result = *this;
    result *= 1 / this->length();
    return result;
}


float *vec3::getValue()
{
    value[0] = x;
    value[1] = y;
    value[2] = z;
    return value;
}


