#include <cmath>

#include "vec4.hpp"

vec4::vec4()
{
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
    w = 0.0f;
}
vec4::vec4(float val)
{
    x = val;
    y = val;
    z = val;
    w = val;
}
vec4::vec4(float a, float b, float c, float d)
{
    x = a;
    y = b;
    z = c;
    w = d;
}

vec4::vec4(const vec3 &vector, float val)
{
    x = vector.x;
    y = vector.y;
    z = vector.z;
    w = val;
}


vec4::vec4(const vec4 &other)
{
    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
}

vec4::~vec4()
{
}

vec4 &vec4::operator=(const vec4 &other)
{
    if (&other == this)
        return *this;

    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
    return *this;
}

vec4 &vec4::operator+=(const vec4 &other)
{
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return *this;
}

vec4 operator+(vec4 lhs, vec4& rhs)
{
    lhs += rhs;
    return lhs;
}

vec4 &vec4::operator*=(float scalar)
{
    x *= scalar;
    y *= scalar;
    z *= scalar;
    w *= scalar;
    return *this;
}

vec4 operator*(vec4 lhs, float scalar)
{
    lhs *= scalar;
    return lhs;
}

vec4 operator*(float scalar, vec4 rhs)
{
    rhs *= scalar;
    return rhs;
}

vec4 operator*(mat4 &matrix, vec4 &vector)
{
    vec4 result(vector);

    result.x = vector.x * matrix.value[0][0] + vector.y * matrix.value[1][0] + vector.z * matrix.value[2][0] + vector.w * matrix.value[3][0];
    result.y = vector.x * matrix.value[0][1] + vector.y * matrix.value[1][1] + vector.z * matrix.value[2][1] + vector.w * matrix.value[3][1];
    result.z = vector.x * matrix.value[0][2] + vector.y * matrix.value[1][2] + vector.z * matrix.value[2][2] + vector.w * matrix.value[3][2];
    result.w = vector.x * matrix.value[0][3] + vector.y * matrix.value[1][3] + vector.z * matrix.value[2][3] + vector.w * matrix.value[3][3];
    return result;
}


float vec4::length()
{
    return sqrt(x * x + y * y + z * z + w * w);
}

float *vec4::getValue()
{
    value[0] = x;
    value[1] = y;
    value[2] = z;
    value[3] = w;
    return value;
}
