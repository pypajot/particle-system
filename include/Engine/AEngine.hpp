#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>
#include <memory>

#include "Shader.hpp"
#include "Camera.hpp"

#define BASE_GRAVITY 1.0f
#define MAX_GRAVITY 2.0f
#define MIN_GRAVITY 0.3f

enum EngineInit
{
    ENGINE_INIT_STATIC,
    ENGINE_INIT_GEN
};

class AEngine
{
    protected:
        std::string _vertexPath;
        const std::string _fragmentPath = "shaders/fragmentShader.fs";
        
        const float _mouseDepth = 2.0f;
        int _particleQty;
        vec3 _gravityPos;
        
        Shader _shader;
    
    public:
        GLuint VBO;
        GLuint VAO;
        
        EngineInit initType;

        bool simulationOn;
        bool gravityOn;
        bool mousePressed;
        
        float gravityStrength;

        Camera camera;
    
        AEngine(int particleQty);
        AEngine(const AEngine &other);
        virtual ~AEngine();

        AEngine &operator=(const AEngine &other);

        void deleteArrays();

        virtual void useShader(float frameTime, float cursorX, float cursorY, float currentHeight) = 0;
        void setGravity(float cursorX, float cursorY, float width, float height);
        void draw() const;

        virtual void reset() = 0;

        virtual void run() = 0;

        void GravityUp();
        void GravityDown();
};