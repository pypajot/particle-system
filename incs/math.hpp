#pragma once

#include "mat4.hpp"
#include "vec3.hpp"

float toRadians(float angle);

mat4 perspective(float fov, float ratio, float near, float far);
mat4 lookAt(vec3 eye, vec3 target, vec3 up);
mat4 rotation(mat4 matrix, float angle, vec3 axis);