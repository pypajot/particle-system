#pragma once

#include "vec4.hpp"
// #include <std::vector>

class mat4
{
    public:
        float value[4][4];

        mat4();
        mat4(float diag);
        mat4(const mat4 &other);
        ~mat4();

        mat4 &operator=(const mat4 &other);

        friend mat4 operator*(mat4 const &lhs, mat4 const &rhs);
        friend mat4 inverse(mat4 matrix);

};