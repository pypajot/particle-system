#pragma once

class vec4;

class vec3
{
    public:
        float x;
        float y;
        float z;

        float value[3];

        constexpr vec3();
        vec3(float val);
        vec3(float x, float y, float z);
        vec3(const vec4 &vector);
        vec3(const vec3 &other);
        ~vec3();
        
        vec3 &operator=(const vec3 &other);

        vec3 &operator+=(const vec3 &other);
        friend vec3 operator+(vec3 lhs, vec3 const &rhs);

        vec3 &operator-=(const vec3 &other);
        friend vec3 operator-(vec3 lhs, vec3 const &rhs);

        vec3 &operator*=(float scalar);
        friend vec3 operator*(vec3 lhs, float scalar);
        friend vec3 operator*(float scalar, vec3 rhs);

        friend float dot(vec3 lhs, vec3 rhs);
        friend vec3 cross(vec3 lhs, vec3 rhs);

        float length() const;
        vec3 normalize();


        float *getValue();
};