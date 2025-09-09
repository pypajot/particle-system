#include <random>

#include "Engine/EngineStatic.hpp"

EngineStatic::EngineStatic() : AEngine()
{
}

EngineStatic::EngineStatic(int particleQuantity) : AEngine(particleQuantity)
{
    initType = ENGINE_INIT_STATIC;

    _vertexPath = STATIC_VERTEX_PATH;
    _shader = Shader(_vertexPath.c_str(), _fragmentPath.c_str());;

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // With a generator initialization, we need the particle position (dim 3) and its speed (dim 3)

    glBufferData(GL_ARRAY_BUFFER, _particleQty * 6 * sizeof(float), 0, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    _worker = WorkerStatic(VBO, _particleQty);

    reset();
}

EngineStatic::EngineStatic(const EngineStatic &other) : AEngine(other)
{
    _worker = other._worker;
}

EngineStatic::~EngineStatic()
{
}

EngineStatic &EngineStatic::operator=(const EngineStatic &other)
{
    if (this == &other)
        return *this;
        
    AEngine::operator=(other);
    _worker = other._worker;
    return *this;
}

/// @brief Reset the engine to a its initial state, but with a cube instead of a sphere
void EngineStatic::resetCube()
{
    _worker.initCube();
    clearGravity();
    camera.resetPosition();
    simulationOn = false;
}

/// @brief Reset the engine to its initial state
void EngineStatic::reset()
{
    _worker.init();
    clearGravity();
    camera.resetPosition();
    simulationOn = false;
}

/// @brief Set the uniform for the shader program and set it to be used
/// @param frameTime The current time for the frame
/// @param cursorX The mouse X coordinate
/// @param cursorY The mouse Y coordinate
/// @param height The window height
void EngineStatic::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(_shader.program, "camera");
    
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, &toScreen.value[0][0]);
    _shader.setFloatUniform("frameTimeX", (1 + sin(frameTime)) / 2);
    _shader.setFloatUniform("frameTimeY", (1 + sin(frameTime + 2 * M_PI / 3)) / 2);
    _shader.setFloatUniform("frameTimeZ", (1 + sin(frameTime - 2 * M_PI / 3)) / 2);
    _shader.setFloatUniform("cursorX", cursorX);
    _shader.setFloatUniform("cursorY", cursorY);
    _shader.setFloatUniform("height", height);
    _shader.setFloatUniform("near", camera.near);
    _shader.setFloatUniform("far", camera.far);
    _shader.setFloatUniform("mouseDepth", _mouseDepth);
    _shader.use();
}

/// @brief Run the simulation for a frame 
void EngineStatic::run()
{
    camera.move();
    _gravity[0].active = mousePressed;
    
    if (!simulationOn)
        return;

    _worker.call(_gravity);
}