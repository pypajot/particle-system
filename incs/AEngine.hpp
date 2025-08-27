#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "Shader.hpp"
#include "Camera.hpp"

class AEngine
{
    public:
        std::string vertexPath = "shaders/vertexShader.vs";
        std::string fragmentPath = "shaders/fragmentShader.fs";

        unsigned int VAO;
        GLuint VBO;

        int particleQty;

        bool simulationOn;
        bool gravityOn;
        glm::vec3 gravityPos;

        Shader shader;
        Camera camera;

        std::string initType;

        AEngine(int particleQty);
        AEngine(AEngine &other);
        virtual ~AEngine();

        AEngine &operator=(AEngine &other);

        void deleteArrays();

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void setGravity(float cursorX, float cursorY);
        void draw();

        virtual void reset() = 0;

        virtual void run() = 0;
};