#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>


class Engine
{
    public:
        GLuint VBO[1]; 
        const char *vertexShader;

        Engine();
        Engine(Engine &other);
        ~Engine();

        void Init();
        int loadShader();
};