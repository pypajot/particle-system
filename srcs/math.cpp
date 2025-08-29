#include <cmath>

#include "math.hpp"

#include <iostream>

float toRadians(float angle)
{
    return angle * (M_PI / 180);
}

mat4 perspective(float fov, float ratio, float near, float far)
{
    double t = tan(fov / 2);

    mat4 result;
    
    result.value[0][0] = 1 / (t * ratio);
    result.value[1][1] = 1 / t;
    result.value[2][2] = -far / (far - near);
    result.value[2][3] = -1;
    result.value[3][2] = -2 * far * near / (far - near);

    return result;
}

mat4 lookAt(vec3 const &eye, vec3 const &target, vec3 const &up)
{
    mat4 result(1.0f);
    vec3 cameraRight, cameraUp, cameraDirection;

    cameraDirection = eye - target;

    cameraDirection = cameraDirection.normalize();
    cameraUp = up;
    cameraRight = cross(cameraUp, cameraDirection);

    cameraUp = cross(cameraDirection, cameraRight);

    cameraRight = cameraRight.normalize();
    cameraUp = cameraUp.normalize();

    result.value[0][0] = cameraRight.x;
    result.value[1][0] = cameraRight.y;
    result.value[2][0] = cameraRight.z;
    result.value[3][0] = -dot(cameraRight, eye);
    result.value[0][1] = cameraUp.x;
    result.value[1][1] = cameraUp.y;
    result.value[2][1] = cameraUp.z;
    result.value[3][1] = -dot(cameraUp, eye);
    result.value[0][2] = cameraDirection.x;
    result.value[1][2] = cameraDirection.y;
    result.value[2][2] = cameraDirection.z;
    result.value[3][2] = -dot(cameraDirection, eye);
    
    return result;
}

mat4 rotation(mat4 const &matrix, float angle, vec3 const &axis)
{
    mat4 rotate(1.0f);

    rotate.value[0][0] = cos(angle) + axis.x * axis.x * (1 - cos(angle));
    rotate.value[1][0] = axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle);
    rotate.value[2][0] = axis.x * axis.z * (1 - cos(angle)) - axis.y * sin(angle);
    rotate.value[0][1] = axis.x * axis.y * (1 - cos(angle)) - axis.z * sin(angle);
    rotate.value[1][1] = cos(angle) + axis.y * axis.y * (1 - cos(angle));
    rotate.value[2][1] = axis.y * axis.z * (1 - cos(angle)) + axis.x * sin(angle);
    rotate.value[0][2] = axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle);
    rotate.value[1][2] = axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle);
    rotate.value[2][2] = cos(angle) + axis.z * axis.z * (1 - cos(angle));

    return matrix * rotate;
}
