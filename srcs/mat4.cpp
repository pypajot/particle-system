#include <stdexcept>

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


mat4 inverse(mat4 matrix)
{
    double s0 = matrix.value[4 * 0 + 0] * matrix.value[4 * 1 + 1] - matrix.value[4 * 1 + 0] * matrix.value[4 * 0 + 1];
    double s1 = matrix.value[4 * 0 + 0] * matrix.value[4 * 1 + 2] - matrix.value[4 * 1 + 0] * matrix.value[4 * 0 + 2];
    double s2 = matrix.value[4 * 0 + 0] * matrix.value[4 * 1 + 3] - matrix.value[4 * 1 + 0] * matrix.value[4 * 0 + 3];
    double s3 = matrix.value[4 * 0 + 1] * matrix.value[4 * 1 + 2] - matrix.value[4 * 1 + 1] * matrix.value[4 * 0 + 2];
    double s4 = matrix.value[4 * 0 + 1] * matrix.value[4 * 1 + 3] - matrix.value[4 * 1 + 1] * matrix.value[4 * 0 + 3];
    double s5 = matrix.value[4 * 0 + 2] * matrix.value[4 * 1 + 3] - matrix.value[4 * 1 + 2] * matrix.value[4 * 0 + 3];

    double c5 = matrix.value[4 * 2 + 2] * matrix.value[4 * 3 + 3] - matrix.value[4 * 3 + 2] * matrix.value[4 * 2 + 3];
    double c4 = matrix.value[4 * 2 + 1] * matrix.value[4 * 3 + 3] - matrix.value[4 * 3 + 1] * matrix.value[4 * 2 + 3];
    double c3 = matrix.value[4 * 2 + 1] * matrix.value[4 * 3 + 2] - matrix.value[4 * 3 + 1] * matrix.value[4 * 2 + 2];
    double c2 = matrix.value[4 * 2 + 0] * matrix.value[4 * 3 + 3] - matrix.value[4 * 3 + 0] * matrix.value[4 * 2 + 3];
    double c1 = matrix.value[4 * 2 + 0] * matrix.value[4 * 3 + 2] - matrix.value[4 * 3 + 0] * matrix.value[4 * 2 + 2];
    double c0 = matrix.value[4 * 2 + 0] * matrix.value[4 * 3 + 1] - matrix.value[4 * 3 + 0] * matrix.value[4 * 2 + 1];

    double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (det == 0)
        throw std::invalid_argument("matrix is not invertible");
    double invdet = 1.0 / det;

    mat4 result;

    result.value[4 * 0 + 0] = ( matrix.value[4 * 1 + 1] * c5 - matrix.value[4 * 1 + 2] * c4 + matrix.value[4 * 1 + 3] * c3) * invdet;
    result.value[4 * 0 + 1] = (-matrix.value[4 * 0 + 1] * c5 + matrix.value[4 * 0 + 2] * c4 - matrix.value[4 * 0 + 3] * c3) * invdet;
    result.value[4 * 0 + 2] = ( matrix.value[4 * 3 + 1] * s5 - matrix.value[4 * 3 + 2] * s4 + matrix.value[4 * 3 + 3] * s3) * invdet;
    result.value[4 * 0 + 3] = (-matrix.value[4 * 2 + 1] * s5 + matrix.value[4 * 2 + 2] * s4 - matrix.value[4 * 2 + 3] * s3) * invdet;

    result.value[4 * 1 + 0] = (-matrix.value[4 * 1 + 0] * c5 + matrix.value[4 * 1 + 2] * c2 - matrix.value[4 * 1 + 3] * c1) * invdet;
    result.value[4 * 1 + 1] = ( matrix.value[4 * 0 + 0] * c5 - matrix.value[4 * 0 + 2] * c2 + matrix.value[4 * 0 + 3] * c1) * invdet;
    result.value[4 * 1 + 2] = (-matrix.value[4 * 3 + 0] * s5 + matrix.value[4 * 3 + 2] * s2 - matrix.value[4 * 3 + 3] * s1) * invdet;
    result.value[4 * 1 + 3] = ( matrix.value[4 * 2 + 0] * s5 - matrix.value[4 * 2 + 2] * s2 + matrix.value[4 * 2 + 3] * s1) * invdet;

    result.value[4 * 2 + 0] = ( matrix.value[4 * 1 + 0] * c4 - matrix.value[4 * 1 + 1] * c2 + matrix.value[4 * 1 + 3] * c0) * invdet;
    result.value[4 * 2 + 1] = (-matrix.value[4 * 0 + 0] * c4 + matrix.value[4 * 0 + 1] * c2 - matrix.value[4 * 0 + 3] * c0) * invdet;
    result.value[4 * 2 + 2] = ( matrix.value[4 * 3 + 0] * s4 - matrix.value[4 * 3 + 1] * s2 + matrix.value[4 * 3 + 3] * s0) * invdet;
    result.value[4 * 2 + 3] = (-matrix.value[4 * 2 + 0] * s4 + matrix.value[4 * 2 + 1] * s2 - matrix.value[4 * 2 + 3] * s0) * invdet;

    result.value[4 * 3 + 0] = (-matrix.value[4 * 1 + 0] * c3 + matrix.value[4 * 1 + 1] * c1 - matrix.value[4 * 1 + 2] * c0) * invdet;
    result.value[4 * 3 + 1] = ( matrix.value[4 * 0 + 0] * c3 - matrix.value[4 * 0 + 1] * c1 + matrix.value[4 * 0 + 2] * c0) * invdet;
    result.value[4 * 3 + 2] = (-matrix.value[4 * 3 + 0] * s3 + matrix.value[4 * 3 + 1] * s1 - matrix.value[4 * 3 + 2] * s0) * invdet;
    result.value[4 * 3 + 3] = ( matrix.value[4 * 2 + 0] * s3 - matrix.value[4 * 2 + 1] * s1 + matrix.value[4 * 2 + 2] * s0) * invdet;
    return result;
}
