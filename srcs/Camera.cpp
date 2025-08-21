#include <glm/ext.hpp>

#include "Camera.hpp"
#include <iostream>

Camera::Camera()
{
    position = glm::vec3(0, 0, 2);
    direction = glm::vec3(0, 0, 1);
}

Camera::Camera(glm::vec3 pos)
{
    position = pos;
    direction = glm::vec3(0, 0, 1);
}

Camera::Camera(Camera &other)
{
    position = other.position;
    direction = other.direction;
    proj = other.proj;
}

Camera::~Camera()
{
}

Camera Camera::operator=(Camera &other)
{
    position = other.position;
    direction = other.direction;
    proj = other.proj;
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

void Camera::moveFront()
{
    position.x += -sin(direction.y) * moveSpeed;
    position.z += -cos(direction.y) * moveSpeed;
}

void Camera::moveBack()
{
    position.x -= -sin(direction.y) * moveSpeed;
    position.z -= -cos(direction.y) * moveSpeed;
}

void Camera::moveLeft()
{
    position.x += -cos(direction.y) * moveSpeed;
    position.z += sin(direction.y) * moveSpeed;
}

void Camera::moveRight()
{
    position.x -= -cos(direction.y) * moveSpeed;
    position.z -= sin(direction.y) * moveSpeed;
}

void Camera::moveUp()
{
    position.y += moveSpeed;
}

void Camera::moveDown()
{
    position.y -= moveSpeed;
}

void Camera::rotateLeft()
{
    direction.y += rotateSpeed;
}

void Camera::rotateRight()
{
    direction.y -= rotateSpeed;
}