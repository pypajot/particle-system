#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>
#include <memory>

#include "Shader.hpp"
#include "Camera.hpp"
#include "Worker/AWorker.hpp"

enum EngineInit
{
    ENGINE_INIT_STATIC,
    ENGINE_INIT_GEN
};

class AEngine
{
    protected:
        EngineInit initType;

        std::string vertexPath;
        const std::string fragmentPath = "shaders/fragmentShader.fs";
        
        int particleQty;
        const float mouseDepth = 2.0f;
        vec3 gravityPos;
        
        Shader shader;
        // std::unique_ptr<AWorker> worker;

    public:
        GLuint VBO;
        GLuint VAO;

        bool simulationOn;
        bool gravityOn;
        bool mousePressed;
        
        Camera camera;
    
        AEngine(int particleQty);
        AEngine(const AEngine &other);
        virtual ~AEngine();

        AEngine &operator=(const AEngine &other);

        void deleteArrays();

        virtual void useShader(float frameTime, float cursorX, float cursorY, float currentHeight) = 0;
        void setGravity(float cursorX, float cursorY, float width, float height);
        void draw();

        virtual void reset() = 0;

        void run() = 0;
};