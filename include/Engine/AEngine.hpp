#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

#include "Shader.hpp"
#include "Camera.hpp"
#include "Gravity.hpp"

#define BASE_MOUSE_DEPTH 2.0f

/// @brief The initialization type of the engine
enum EngineInit
{
    /// @brief The engine uses a static initialization
    ENGINE_INIT_STATIC,
    /// @brief The engine uses a generator for its initialization
    ENGINE_INIT_GEN
};

/// @brief Base engine class from which are derived the actual engines, based on their initlaization
class AEngine
{
    protected:
        /// @brief Path to the vertex shader
        std::string _vertexPath;
        /// @brief Path to the fragment shader
        const std::string _fragmentPath = "shaders/fragmentShader.fs";
        
        /// @brief The mouse depth that will be used in world coordinates
        const float _mouseDepth = BASE_MOUSE_DEPTH;
        /// @brief The number of particle used by the engine
        int _particleQty;

        /// @brief All gravity points present in the engine
        /// @note The element at index 0 always exists and represents the mouse 
        std::vector<Gravity> _gravity;
        
        /// @brief The shader used for display in the engine
        Shader _shader;
    
        vec3 _cursorToWorld(float cursorX, float cursorY, float width, float height) const;

    public:
        /// @brief The buffer vertex object
        GLuint VBO;
        /// @brief The vertex array object
        GLuint VAO;
        
        /// @brief The initialization type of the engine
        EngineInit initType;

        /// @brief The status of the simulation
        bool simulationOn;
        /// @brief The status of the mouse
        bool mousePressed;
        
        /// @brief The camera object used for the display
        Camera camera;
    
        AEngine();
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