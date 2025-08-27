#include <glm/ext.hpp>

#include "Camera.hpp"
#include <iostream>

Camera::Camera()
{
    position = glm::vec3(0, 0, 2);
    direction = glm::vec3(0, 0, 1);

    moveFrontBack = 0;
    moveLeftRight = 0;
    moveUpDown = 0;
    rotateLeftRight = 0;
}

Camera::Camera(glm::vec3 pos)
{
    position = pos;
    direction = glm::vec3(0, 0, 1);

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

void Camera::computeProjectionMatrix(float height, float width)
{
    proj = glm::perspective(fov, width / height, near, far);
}

glm::mat4 Camera::coordToScreenMatrix()
{
    glm::vec3 cameraFront(-sin(direction.y), 0.0f, -cos(direction.y));
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

    glm::mat4 toCamera = glm::lookAt(position, position + cameraFront, cameraUp);

    return proj * toCamera;
}

void Camera::move()
{
    direction.y += rotateSpeed * rotateLeftRight;

    glm::vec4 movement(-moveLeftRight, moveUpDown, -moveFrontBack, 1.0f);
    glm::mat4 rotate(1.0f);

    movement /= movement.length();
    rotate = glm::rotate(rotate, direction.y, glm::vec3(0.0f, 1.0f, 0.0f));
    movement = movement * rotate;
    position += movement * moveSpeed;
}

void Camera::resetPosition()
{
    position = glm::vec3(0, 0, 2);
    direction = glm::vec3(0, 0, 1);
}