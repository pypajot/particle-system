#pragma once

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "math/mat4.hpp"

class Camera
{
    private:
        const float fov = toRadians(60.0f);
        const float near = 0.1f;
        const float far = 100.0f;
        mat4 proj;
        
        const float rotateSpeed = 0.01f;
        const float moveSpeed = 0.01f;

        vec3 position;
        vec3 direction;

    public:
        int moveFrontBack;
        int moveLeftRight;
        int moveUpDown;
        int rotateLeftRight;

        Camera();
        Camera(vec3 postiion);
        Camera(Camera &other);
        ~Camera();

        Camera operator=(Camera &other);

        void move();
        void resetPosition();
        
        void computeProjectionMatrix(float height, float width);
        mat4 coordToScreenMatrix() const;


};