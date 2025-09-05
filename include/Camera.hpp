#pragma once

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "math/mat4.hpp"

class Camera
{
    private:
        mat4 _proj;
        
        const float _rotateSpeed = 0.01f;
        const float _moveSpeed = 0.01f;
        
        vec3 _position;
        vec3 _direction;
    
    public:
        const float fov = toRadians(60.0f);
        const float near = 0.1f;
        const float far = 100.0f;

        int moveFrontBack;
        int moveLeftRight;
        int moveUpDown;
        int rotateLeftRight;

        Camera();
        Camera(vec3 postiion);
        Camera(const Camera &other);
        ~Camera();

        Camera operator=(const Camera &other);

        void move();
        void resetPosition();
        
        void computeProjectionMatrix(float height, float width);
        mat4 coordToScreenMatrix() const;


};