#pragma once

#include "math.hpp"
#include "vec3.hpp"
#include "mat4.hpp"

class Camera
{
    public:
        vec3 position;
        vec3 direction;
        const float fov = toRadians(60.0f);
        const float near = 0.1f;
        const float far = 100.0f;
        mat4 proj;

        const float rotateSpeed = 0.01f;
        const float moveSpeed = 0.01f;

        Camera();
        Camera(vec3 postiion);
        Camera(Camera &other);
        ~Camera();

        Camera operator=(Camera &other);

        void computeProjectionMatrix(float height, float width);
        mat4 coordToScreenMatrix();

        int moveFrontBack;
        int moveLeftRight;
        int moveUpDown;
        int rotateLeftRight;

        void move();
        void resetPosition();

};