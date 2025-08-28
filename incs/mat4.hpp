#pragma once


class mat4
{
    public:
        float value[16];

        mat4();
        mat4(float diag);
        mat4(const mat4 &other);
        ~mat4();

        mat4 &operator=(const mat4 &other);


        friend mat4 operator*(mat4 &lhs, mat4 &rhs);
};