#pragma once

class mat4;
class vec3;

class vec4
{
    public:
        float x;
        float y;
        float z;
        float w;

        float value[4];

        vec4();
        vec4(float val);
        vec4(float x, float y, float z, float w);
        vec4(const vec3 &other, float w);
        vec4(const vec4 &vector);
        ~vec4();
        
        vec4 &operator=(const vec4 &other);

        vec4 &operator+=(const vec4 &other);
        friend vec4 operator+(vec4 lhs, vec4& rhs);

        vec4 &operator*=(float scalar);
        friend vec4 operator*(vec4 lhs, float scalar);
        friend vec4 operator*(float scalar, vec4 rhs);

        friend vec4 operator*(const mat4 &matrix, const vec4 &vector);

        float length() const;

        float *getValue();
};