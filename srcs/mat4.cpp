#include "mat4.hpp"

mat4::mat4()
{
    for(int i = 0; i < 16; i++)
    {
        value[i] = 0.0f;
    }
}

mat4::mat4(float diag)
{
    value[0] = diag;
    value[5] = diag;
    value[10] = diag;
    value[15] = diag;
}

mat4::mat4(const mat4 &other)
{
    for(int i = 0; i < 16; i++)
    {
        value[i] = other.value[i];
    }
}

mat4::~mat4()
{
}

mat4 &mat4::operator=(const mat4 &other)
{
    if (&other == this)
        return *this;

    for(int i = 0; i < 16; i++)
    {
        value[i] = other.value[i];
    }
    return *this;
}


mat4 operator*(mat4 &lhs, mat4 &rhs)
{
    mat4 result;
    int x;
    int y;

    for(int i = 0; i < 16; i++)
    {
        x = i % 4;
        y = i / 4;
        for (int j = 0; j < 4; j++)
        {
            result.value[i] += lhs.value[j + 4 * y] * rhs.value[4 * j + x];
        }
    }
    return result; 
}