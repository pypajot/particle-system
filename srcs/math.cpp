#include <cmath>

#include "math.hpp"

#include <iostream>

float toRadians(float angle)
{
    return angle * (M_PI / 180);
}

mat4 perspective(float fov, float ratio, float near, float far)
{
    double height, width;
    width = tan(fov / 2) * near;
    height = width / ratio;
    mat4 result;
    
    result.value[0][0] = near / width;
    result.value[1][1] = near / height;
    result.value[2][2] = -far / (far - near);
    result.value[2][3] = -far * near / (far - near);
    result.value[3][2] = -1;

    return result;
}

mat4 lookAt(vec3 eye, vec3 target, vec3 up)
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
    result.value[0][1] = cameraRight.y;
    result.value[0][2] = cameraRight.z;
    result.value[0][3] = -dot(cameraRight, eye);
    result.value[1][0] = cameraUp.x;
    result.value[1][1] = cameraUp.y;
    result.value[1][2] = cameraUp.z;
    result.value[1][3] = -dot(cameraUp, eye);
    result.value[2][0] = cameraDirection.x;
    result.value[2][1] = cameraDirection.y;
    result.value[2][2] = cameraDirection.z;
    result.value[2][3] = -dot(cameraDirection, eye);
    
    return result;
}

mat4 rotation(mat4 matrix, float angle, vec3 axis)
{
    mat4 rotate(1.0f);

    rotate.value[0][0] = cos(angle) + axis.x * axis.x * (1 - cos(angle));
    rotate.value[0][1] = axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle);
    rotate.value[0][2] = axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle);
    rotate.value[1][0] = axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle);
    rotate.value[1][1] = cos(angle) + axis.y * axis.y * (1 - cos(angle));
    rotate.value[1][2] = axis.y * axis.z * (1 - cos(angle)) + axis.x * sin(angle);
    rotate.value[2][0] = axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle);
    rotate.value[2][1] = axis.y * axis.z * (1 - cos(angle)) + axis.x * sin(angle);
    rotate.value[2][2] = cos(angle) + axis.z * axis.z * (1 - cos(angle));

    return matrix * rotate;
}
