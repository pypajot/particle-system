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
}

Camera::~Camera()
{
}

Camera Camera::operator=(Camera &other)
{
    position = other.position;
    direction = other.direction;
    return *this;
}


glm::mat4 Camera::coordToScreenMatrix(float height, float width)
{
    glm::mat4 projection(1.0f);
    projection = glm::perspective(fov, width / height, near, far);
    glm::mat4 toCamera(1.0f);
    glm::mat4 tr(1.0f);
    (void)projection;
    toCamera = glm::rotate(toCamera, direction.x, glm::vec3(1.0f, 0.0f, 0.0f));
    toCamera = glm::rotate(toCamera, direction.y, glm::vec3(0.0f, 1.0f, 0.0f));
    tr = glm::translate(tr, -position);
    return projection * tr * toCamera;
}