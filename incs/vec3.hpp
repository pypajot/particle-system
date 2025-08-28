#pragma once

#include "vec4.hpp"

class vec3
{
    public:
        float x;
        float y;
        float z;

        float value[3];

        vec3();
        vec3(float val);
        vec3(float x, float y, float z);
        vec3(const vec4 &vector);
        vec3(const vec3 &other);
        ~vec3();
        

        vec3 &operator=(const vec3 &other);
        vec3 &operator+=(const vec3 &other);
        friend vec3 operator+(vec3 lhs, vec3& rhs);

        float length();


        float *getValue();
};