#include <cmath>

#include "transform.hpp"
#include "math/vec4.hpp"
#include "Camera.hpp"

#include <iostream>

/// @brief Construct a Camera object at default position (0, 0, 2) and direction (0, 0, 1)
Camera::Camera()
{
    _position = vec3(0, 0, 2);
    _direction = vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

/// @brief Constrcut a Camera object with default direction (0, 0, 1)
/// @param pos The position at which the camera will be set on creation 
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

/// @brief Move the cameras depending on its position and direction
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

/// @brief Reset teh camera position and direction
void Camera::resetPosition()
{
    _position = vec3(0, 0, 2);
    _direction = vec3(0, 0, 1);
}

/// @brief Computes the projection matrix
/// @note Need to be called when the window changes size
/// @param height The height of the window
/// @param width The width
void Camera::computeProjectionMatrix(float height, float width)
{
    _proj = perspective(fov, width / height, near, far);
}

/// @brief Computes the matrix to convert world coordinates to screen coordinate sfor use by openGL
/// @return The ematrix
mat4 Camera::coordToScreenMatrix() const
{
    vec3 cameraFront(-std::sin(_direction.y), 0.0f, -std::cos(_direction.y));
    vec3 cameraUp(0.0f, 1.0f, 0.0f);

    mat4 toCamera = lookAt(_position, _position + cameraFront, cameraUp);

    return _proj * toCamera;
}
