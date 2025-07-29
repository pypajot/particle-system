#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>


class Engine
{
    public:
        GLuint VBO[1];
        std::string vertexPath = "shaders/vertexShader.vs";
        std::string vertexShader;

        Engine();
        Engine(Engine &other);
        ~Engine();

        void Init();
        int loadShader();
};