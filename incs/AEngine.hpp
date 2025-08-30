#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>

#include "Shader.hpp"
#include "Camera.hpp"

class AEngine
{
    public:
        const std::string vertexPath = "shaders/vertexShader.vs";
        const std::string fragmentPath = "shaders/fragmentShader.fs";

        const float mouseDepth = 2.0f;

        unsigned int VAO;
        GLuint VBO;

        int particleQty;

        bool simulationOn;
        bool gravityOn;
        vec3 gravityPos;

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