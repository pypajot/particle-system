#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "Shader.hpp"
#include "Camera.hpp"

class Engine
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

        Engine(int particleQty);
        Engine(Engine &other);
        ~Engine();

        Engine operator=(Engine &other);

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void draw();
        void initSphere();
        void initCube();
};