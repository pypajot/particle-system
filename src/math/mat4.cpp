#include <stdexcept>

#include "math/mat4.hpp"

mat4::mat4()
{
    for(int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            value[j][i] = 0.0f;
}

mat4::mat4(float diag)
{
    for(int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
        {   
            if (j == i)
                value[j][i] = diag;
            else
                value[j][i] = 0.0f;
        }
}

mat4::mat4(const mat4 &other)
{

    for(int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            value[j][i] = other.value[j][i];
}

mat4::~mat4()
{
}

mat4 &mat4::operator=(const mat4 &other)
{
    if (&other == this)
        return *this;

    for(int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            value[j][i] = other.value[j][i];

    return *this;
}


mat4 operator*(mat4 const &lhs, mat4 const &rhs)
{
    mat4 result;

    for(int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            for (int k = 0; k < 4; k++)
            {
                result.value[i][j] += lhs.value[k][j] * rhs.value[i][k];
            }

    return result; 
}


mat4 inverse(mat4 matrix)
{
    double s0 = matrix.value[0][0] * matrix.value[1][1] - matrix.value[0][1] * matrix.value[1][0];
    double s1 = matrix.value[0][0] * matrix.value[2][1] - matrix.value[0][1] * matrix.value[2][0];
    double s2 = matrix.value[0][0] * matrix.value[3][1] - matrix.value[0][1] * matrix.value[3][0];
    double s3 = matrix.value[1][0] * matrix.value[2][1] - matrix.value[1][1] * matrix.value[2][0];
    double s4 = matrix.value[1][0] * matrix.value[3][1] - matrix.value[1][1] * matrix.value[3][0];
    double s5 = matrix.value[2][0] * matrix.value[3][1] - matrix.value[2][1] * matrix.value[3][0];

    double c5 = matrix.value[2][2] * matrix.value[3][3] - matrix.value[2][3] * matrix.value[3][2];
    double c4 = matrix.value[1][2] * matrix.value[3][3] - matrix.value[1][3] * matrix.value[3][2];
    double c3 = matrix.value[1][2] * matrix.value[2][3] - matrix.value[1][3] * matrix.value[2][2];
    double c2 = matrix.value[0][2] * matrix.value[3][3] - matrix.value[0][3] * matrix.value[3][2];
    double c1 = matrix.value[0][2] * matrix.value[2][3] - matrix.value[0][3] * matrix.value[2][2];
    double c0 = matrix.value[0][2] * matrix.value[1][3] - matrix.value[0][3] * matrix.value[1][2];

    double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if (det == 0)
        throw std::invalid_argument("matrix is not invertible");
    double invdet = 1.0 / det;

    mat4 result;

    result.value[0][0] = ( matrix.value[1][1] * c5 - matrix.value[2][1] * c4 + matrix.value[3][1] * c3) * invdet;
    result.value[1][0] = (-matrix.value[1][0] * c5 + matrix.value[2][0] * c4 - matrix.value[3][0] * c3) * invdet;
    result.value[2][0] = ( matrix.value[1][3] * s5 - matrix.value[2][3] * s4 + matrix.value[3][3] * s3) * invdet;
    result.value[3][0] = (-matrix.value[1][2] * s5 + matrix.value[2][2] * s4 - matrix.value[3][2] * s3) * invdet;

    result.value[0][1] = (-matrix.value[0][1] * c5 + matrix.value[2][1] * c2 - matrix.value[3][1] * c1) * invdet;
    result.value[1][1] = ( matrix.value[0][0] * c5 - matrix.value[2][0] * c2 + matrix.value[3][0] * c1) * invdet;
    result.value[2][1] = (-matrix.value[0][3] * s5 + matrix.value[2][3] * s2 - matrix.value[3][3] * s1) * invdet;
    result.value[3][1] = ( matrix.value[0][2] * s5 - matrix.value[2][2] * s2 + matrix.value[3][2] * s1) * invdet;

    result.value[0][2] = ( matrix.value[0][1] * c4 - matrix.value[1][1] * c2 + matrix.value[3][1] * c0) * invdet;
    result.value[1][2] = (-matrix.value[0][0] * c4 + matrix.value[1][0] * c2 - matrix.value[3][0] * c0) * invdet;
    result.value[2][2] = ( matrix.value[0][3] * s4 - matrix.value[1][3] * s2 + matrix.value[3][3] * s0) * invdet;
    result.value[3][2] = (-matrix.value[0][2] * s4 + matrix.value[1][2] * s2 - matrix.value[3][2] * s0) * invdet;

    result.value[0][3] = (-matrix.value[0][1] * c3 + matrix.value[1][1] * c1 - matrix.value[2][1] * c0) * invdet;
    result.value[1][3] = ( matrix.value[0][0] * c3 - matrix.value[1][0] * c1 + matrix.value[2][0] * c0) * invdet;
    result.value[2][3] = (-matrix.value[0][3] * s3 + matrix.value[1][3] * s1 - matrix.value[2][3] * s0) * invdet;
    result.value[3][3] = ( matrix.value[0][2] * s3 - matrix.value[1][2] * s1 + matrix.value[2][2] * s0) * invdet;
    return result;
}
