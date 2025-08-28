#include <cmath>

#include "math.hpp"

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

    result.value[0] = near / width;
    result.value[5] = near / height;
    result.value[10] = -far / (far - near);
    result.value[11] = -far * near / (far - near);
    result.value[13] = -1;

    return result;
}

mat4 lookAt(vec3 eye, vec3 target, vec3 up)
{
    mat4 result(1.0f);
    vec3 newX, newY, newZ;

    newZ = eye - target;
    newZ = newZ.normalize();
    newY = up;
    newZ = cross(newY, newZ);
    newY = cross(newZ, newX);
    newX = newX.normalize();
    newY = newY.normalize();

    result.value[0 + 0 * 4] = newX.x;
    result.value[1 + 0 * 4] = newX.y;
    result.value[2 + 0 * 4] = newX.z;
    result.value[3 + 0 * 4] = -dot(newX, eye);
    result.value[0 + 1 * 4] = newY.x;
    result.value[1 + 1 * 4] = newY.y;
    result.value[2 + 1 * 4] = newY.z;
    result.value[3 + 1 * 4] = -dot(newY, eye);
    result.value[0 + 2 * 4] = newZ.x;
    result.value[1 + 2 * 4] = newZ.y;
    result.value[2 + 2 * 4] = newZ.z;
    result.value[3 + 2 * 4] = -dot(newZ, eye);
    
    return result;
}

mat4 rotation(mat4 matrix, float angle, vec3 axis)
{
    mat4 rotate(1.0f);

    rotate.value[0] = cos(angle) + axis.x * axis.x * (1 - cos(angle));
    rotate.value[0] = axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle);
    rotate.value[0] = axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle);
    rotate.value[0] = axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle);
    rotate.value[0] = cos(angle) + axis.y * axis.y * (1 - cos(angle));
    rotate.value[0] = axis.y * axis.z * (1 - cos(angle)) + axis.x * sin(angle);
    rotate.value[0] = axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle);
    rotate.value[0] = axis.y * axis.z * (1 - cos(angle)) + axis.x * sin(angle);
    rotate.value[0] = cos(angle) + axis.z * axis.z * (1 - cos(angle));

    return matrix * rotate;
}
