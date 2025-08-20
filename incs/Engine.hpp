#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "Shader.hpp"
#include "Camera.hpp"

class Engine
{
    public:
        unsigned int VAO;
        GLuint VBO;
        std::string vertexPath = "shaders/vertexShader.vs";
        std::string fragmentPath = "shaders/fragmentShader.fs";
        int particleQty;

        Shader shader;
        Camera camera;

        Engine(int particleQty);
        Engine(Engine &other);
        ~Engine();

        Engine operator=(Engine &other);

        void useShader(float frameTime, float cursorX, float cursorY, float currentHeight);
        void draw();
};