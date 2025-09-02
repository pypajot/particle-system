#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>

#include "Shader.hpp"
#include "Camera.hpp"
#include "CudaWorker.hpp"

class AEngine
{
    public:
        std::string vertexPath;
        const std::string fragmentPath = "shaders/fragmentShader.fs";

        const float mouseDepth = 2.0f;

        unsigned int VAO;
        GLuint VBO;

        int particleQty;

        bool simulationOn;
        bool gravityOn;
        bool mousePressed;
        vec3 gravityPos;

        Shader shader;
        Camera camera;

        std::string initType;

        CudaWorker gravity;

        AEngine(int particleQty);
        // AEngine(AEngine &other);
        virtual ~AEngine();

        // AEngine &operator=(AEngine &other);

        void deleteArrays();

        virtual void useShader(float frameTime, float cursorX, float cursorY, float currentHeight) = 0;
        void setGravity(float cursorX, float cursorY);
        void draw();

        virtual void reset() = 0;

        virtual void run() = 0;
};