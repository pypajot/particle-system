#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

#include "Camera.hpp"

#include <iostream>

/// @brief Construct a Camera object at default position (0, 0, 2) and direction (0, 0, 1)
Camera::Camera()
{
    _position = glm::vec3(0, 0, 2);
    _direction = glm::vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

/// @brief Constrcut a Camera object with default direction (0, 0, 1)
/// @param pos The position at which the camera will be set on creation 
Camera::Camera(const glm::vec3 &pos)
{
    _position = pos;
    _direction = glm::vec3(0, 0, 1);

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
    glm::vec3 movement(-moveLeftRight, moveUpDown, -moveFrontBack);
    glm::mat4 rotate(1.0f);
    
    if (movement.length() == 0)
        return;

    movement /= movement.length();
    rotate = glm::rotate(rotate, _direction.y, glm::vec3(0.0f, 1.0f, 0.0f));
    movement = rotate * glm::vec4(movement, 1.0f);
    _position += movement * _moveSpeed;
}

/// @brief Reset teh camera position and direction
void Camera::resetPosition()
{
    _position = glm::vec3(0, 0, 2);
    _direction = glm::vec3(0, 0, 1);
}

/// @brief Computes the projection matrix
/// @note Need to be called when the window changes size
/// @param height The height of the window
/// @param width The width
void Camera::computeProjectionMatrix(float height, float width)
{
    _proj = glm::perspective(fov, width / height, near, far);
}

/// @brief Computes the matrix to convert world coordinates to screen coordinate sfor use by openGL
/// @return The matrix
glm::mat4 Camera::coordToScreenMatrix() const
{
    glm::vec3 cameraFront(-std::sin(_direction.y), 0.0f, -std::cos(_direction.y));
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

    glm::mat4 toCamera = glm::lookAt(_position, _position + cameraFront, cameraUp);

    return _proj * toCamera;
}
