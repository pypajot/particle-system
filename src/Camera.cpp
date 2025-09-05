#include <cmath>

#include "transform.hpp"
#include "math/vec4.hpp"
#include "Camera.hpp"

#include <iostream>

Camera::Camera()
{
    _position = vec3(0, 0, 2);
    _direction = vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

Camera::Camera(vec3 pos)
{
    _position = pos;
    _direction = vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

Camera::Camera(const Camera &other)
{
    _position = other._position;
    _direction = other._direction;
    _proj = other._proj;

    moveFrontBack = other.moveFrontBack;
    moveLeftRight = other.moveLeftRight;
    moveUpDown = other.moveUpDown;
    rotateLeftRight = other.rotateLeftRight;
}

Camera::~Camera()
{
}

Camera Camera::operator=(const Camera &other)
{
    _position = other._position;
    _direction = other._direction;
    _proj = other._proj;

    moveFrontBack = other.moveFrontBack;
    moveLeftRight = other.moveLeftRight;
    moveUpDown = other.moveUpDown;
    rotateLeftRight = other.rotateLeftRight;

    return *this;
}

void Camera::move()
{
    _direction.y += _rotateSpeed * rotateLeftRight;
    vec3 movement(-moveLeftRight, moveUpDown, -moveFrontBack);
    mat4 rotate(1.0f);
    
    if (movement.length() == 0)
    return;
    movement *= 1 / movement.length();
    rotate = rotation(rotate, -_direction.y, vec3(0.0f, 1.0f, 0.0f));
    movement = rotate * vec4(movement, 1.0f);
    _position += movement * _moveSpeed;
}

void Camera::resetPosition()
{
    _position = vec3(0, 0, 2);
    _direction = vec3(0, 0, 1);
}

void Camera::computeProjectionMatrix(float height, float width)
{
    _proj = perspective(fov, width / height, near, far);
}

mat4 Camera::coordToScreenMatrix() const
{
    vec3 cameraFront(-std::sin(_direction.y), 0.0f, -std::cos(_direction.y));
    vec3 cameraUp(0.0f, 1.0f, 0.0f);

    mat4 toCamera = lookAt(_position, _position + cameraFront, cameraUp);

    return _proj * toCamera;
}
