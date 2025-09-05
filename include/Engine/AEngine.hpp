#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

#include "Shader.hpp"
#include "Camera.hpp"
#include "Gravity.hpp"

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

        std::vector<Gravity> _gravity;
        
        Shader _shader;
    
        vec3 AEngine::_cursorToWorld(float cursorX, float cursorY, float width, float height) const;

    public:
        GLuint VBO;
        GLuint VAO;
        
        EngineInit initType;

        bool simulationOn;
        bool mousePressed;
        
        Camera camera;
    
        AEngine(int particleQty);
        AEngine(const AEngine &other);
        virtual ~AEngine();

        AEngine &operator=(const AEngine &other);

        void deleteArrays();

        virtual void useShader(float frameTime, float cursorX, float cursorY, float currentHeight) = 0;
        void draw() const;
        
        virtual void reset() = 0;
        
        virtual void run() = 0;
        
        void setMouseGravity(float cursorX, float cursorY, float width, float height);
        void addGravity(float cursorX, float cursorY, float width, float height);
        void clearGravity();
        void allGravityUp();
        void allGravityDown();
};