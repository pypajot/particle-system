#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>

class Camera
{
    public:
        glm::vec3 position;
        glm::vec3 direction;
        const float fov = glm::radians(60.0f);
        const float near = 0.1f;
        const float far = 100.0f;
        glm::mat4 proj;

        const float rotateSpeed = 0.01f;
        const float moveSpeed = 0.01f;

        Camera();
        Camera(glm::vec3 postiion);
        Camera(Camera &other);
        ~Camera();

        Camera operator=(Camera &other);

        void computeProjectionMatrix(float height, float width);
        glm::mat4 coordToScreenMatrix();

        void moveFront();
        void moveBack();
        void moveLeft();
        void moveRight();
        void moveUp();
        void moveDown();
        void rotateLeft();
        void rotateRight();
};