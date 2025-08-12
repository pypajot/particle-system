#include <glm/ext.hpp>

#include "Camera.hpp"

Camera::Camera()
{
    position = glm::vec3(0, 0, 1);
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
    glm::vec3 cameraFront(-glm::sin(direction.y), 0.0f, -cos(direction.y));
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
    glm::mat4 toCamera = glm::lookAt(position, position + cameraFront, cameraUp);
    
    glm::mat4 tr(1.0f);
    tr = glm::translate(tr, -position);

    return proj * tr * toCamera;
}