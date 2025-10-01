#pragma once

#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

/// @brief The class dealing with the movement of the camera but also the projection matrix and coordinate change from screen to world
class Camera
{
    private:
        /// @brief The current projection matrix
        glm::mat4 _proj;
        
        /// @brief The camera rotation speed in radians
        const float _rotateSpeed = 0.01f;
        /// @brief The camera movement speed in world coordinates
        const float _moveSpeed = 0.05f;
        
        /// @brief The camera posistion
        glm::vec3 _position;
        /// @brief The camera direction
        glm::vec3 _direction;
    
    public:
        /// @brief The fov of the camera
        const float fov = glm::radians(60.0f);
        /// @brief the distance to the near clipping plane
        const float near = 0.1f;
        /// @brief the distance to the far clipping plane
        const float far = 100.0f;

        /// @brief Integer to represent the front / back movement of the camera. A positive value means the camera is moving forward, a negative one backward
        int moveFrontBack;
        /// @brief Integer to represent the left / right movement of the camera. A positive value means the camera is moving left, a negative one right
        int moveLeftRight;
        /// @brief Integer to represent the up / down movement of the camera. A positive value means the camera is moving up, a negative one down
        int moveUpDown;
        /// @brief Integer to represent the left / right rotation of the camera. A positive value means the camera is rotating left, a negative one right
        int rotateLeftRight;

        Camera();
        Camera(const glm::vec3 &position);
        Camera(const Camera &other);
        ~Camera();

        Camera operator=(const Camera &other);

        void move();
        void resetPosition();
        
        void computeProjectionMatrix(float height, float width);
        glm::mat4 coordToScreenMatrix() const;
};