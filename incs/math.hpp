#pragma once

#include "mat4.hpp"
#include "vec3.hpp"

float toRadians(float angle);

mat4 perspective(float fov, float ratio, float near, float far);
mat4 lookAt(vec3 const &eye, vec3 const &target, vec3 const &up);
mat4 rotation(mat4 const &matrix, float angle, vec3 const &axis);