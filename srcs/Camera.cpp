#include <cmath>

#include "math.hpp"
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

void Camera::computeProjectionMatrix(float height, float width)
{
    proj = perspective(fov, width / height, near, far);
}

mat4 Camera::coordToScreenMatrix()
{
    vec3 cameraFront(-std::sin(direction.y), 0.0f, -std::cos(direction.y));
    vec3 cameraUp(0.0f, 1.0f, 0.0f);

    mat4 toCamera = lookAt(position, position + cameraFront, cameraUp);
    std::cout << toCamera.value[0][0] << " " << toCamera.value[0][1] << " " << toCamera.value[0][2] << " " << toCamera.value[0][3] << "\n";
    std::cout << toCamera.value[1][0] << " " << toCamera.value[1][1] << " " << toCamera.value[1][2] << " " << toCamera.value[1][3] << "\n";
    std::cout << toCamera.value[2][0] << " " << toCamera.value[2][1] << " " << toCamera.value[2][2] << " " << toCamera.value[2][3] << "\n";
    std::cout << toCamera.value[3][0] << " " << toCamera.value[3][1] << " " << toCamera.value[3][2] << " " << toCamera.value[3][3] << "\n";
    std::cout << proj.value[0][0] << " " << proj.value[0][1] << " " << proj.value[0][2] << " " << proj.value[0][3] << "\n";
    std::cout << proj.value[1][0] << " " << proj.value[1][1] << " " << proj.value[1][2] << " " << proj.value[1][3] << "\n";
    std::cout << proj.value[2][0] << " " << proj.value[2][1] << " " << proj.value[2][2] << " " << proj.value[2][3] << "\n";
    std::cout << proj.value[3][0] << " " << proj.value[3][1] << " " << proj.value[3][2] << " " << proj.value[3][3] << "\n";

    return proj * toCamera;
}

void Camera::move()
{
    direction.y += rotateSpeed * rotateLeftRight;

    vec4 movement(-moveLeftRight, moveUpDown, -moveFrontBack, 1.0f);
    mat4 rotate(1.0f);

    movement *= 1 / movement.length();
    rotate = rotation(rotate, direction.y, vec3(0.0f, 1.0f, 0.0f));
    std::cout << rotate.value[0][0] << " " << rotate.value[0][1] << " " << rotate.value[0][2] << " " << rotate.value[0][3] << "\n";
    std::cout << rotate.value[1][0] << " " << rotate.value[1][1] << " " << rotate.value[1][2] << " " << rotate.value[1][3] << "\n";
    std::cout << rotate.value[2][0] << " " << rotate.value[2][1] << " " << rotate.value[2][2] << " " << rotate.value[2][3] << "\n";
    std::cout << rotate.value[3][0] << " " << rotate.value[3][1] << " " << rotate.value[3][2] << " " << rotate.value[3][3] << "\n";
    movement = rotate * movement;
    position += movement * moveSpeed;
}

void Camera::resetPosition()
{
    position = vec3(0, 0, 2);
    direction = vec3(0, 0, 1);
}