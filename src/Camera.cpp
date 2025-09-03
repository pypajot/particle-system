#include <cmath>

#include "math/transform.hpp"
#include "math/vec4.hpp"
#include "Camera.hpp"

#include <iostream>

Camera::Camera()
{
    position = vec3(0, 0, 2);
    direction = vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

Camera::Camera(vec3 pos)
{
    position = pos;
    direction = vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

Camera::Camera(Camera &other)
{
    position = other.position;
    direction = other.direction;
    proj = other.proj;

    moveFrontBack = other.moveFrontBack;
    moveLeftRight = other.moveLeftRight;
    moveUpDown = other.moveUpDown;
    rotateLeftRight = other.rotateLeftRight;
}

Camera::~Camera()
{
}

Camera Camera::operator=(Camera &other)
{
    position = other.position;
    direction = other.direction;
    proj = other.proj;

    moveFrontBack = other.moveFrontBack;
    moveLeftRight = other.moveLeftRight;
    moveUpDown = other.moveUpDown;
    rotateLeftRight = other.rotateLeftRight;

    return *this;
}

void Camera::move()
{
    direction.y += rotateSpeed * rotateLeftRight;
    vec3 movement(-moveLeftRight, moveUpDown, -moveFrontBack);
    mat4 rotate(1.0f);
    
    if (movement.length() == 0)
    return;
    movement *= 1 / movement.length();
    rotate = rotation(rotate, direction.y, vec3(0.0f, 1.0f, 0.0f));
    movement = rotate * vec4(movement, 1.0f);
    position += movement * moveSpeed;
}

void Camera::resetPosition()
{
    position = vec3(0, 0, 2);
    direction = vec3(0, 0, 1);
}

void Camera::computeProjectionMatrix(float height, float width)
{
    proj = perspective(fov, width / height, near, far);
}

mat4 Camera::coordToScreenMatrix() const
{
    vec3 cameraFront(-std::sin(direction.y), 0.0f, -std::cos(direction.y));
    vec3 cameraUp(0.0f, 1.0f, 0.0f);

    mat4 toCamera = lookAt(position, position + cameraFront, cameraUp);

    return proj * toCamera;
}
