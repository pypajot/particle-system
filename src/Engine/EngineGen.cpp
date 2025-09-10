#include <cmath>
#include <iostream>

#include "Engine/EngineGen.hpp"

EngineGen::EngineGen() : AEngine()
{
}

EngineGen::EngineGen(int particleQuantity, int ttl) : AEngine(particleQuantity), _timeToLive(ttl)
{
    initType = ENGINE_INIT_GEN;    

    _vertexPath = GEN_VERTEX_PATH;
    _shader = Shader(_vertexPath.c_str(), _fragmentPath.c_str());
    _particlePerFrame = BASE_PPF;

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // With a generator initialization, we need the particle position (dim 3), its speed (dim 3), and how for long it has been alive (dim 1) 

    glBufferData(GL_ARRAY_BUFFER, _particleQty * 7 * sizeof(float), 0, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    _worker = WorkerGen(VBO, _particleQty);

    reset();
}

EngineGen::EngineGen(const EngineGen &other) : AEngine(other)
{
    _particlePerFrame = other._particlePerFrame;
    generatorOn = other.generatorOn;
    _worker = other._worker;
}

EngineGen::~EngineGen()
{
}

EngineGen &EngineGen::operator=(const EngineGen &other)
{
    if (this == &other)
        return *this;
        
    AEngine::operator=(other);
    _particlePerFrame = other._particlePerFrame;
    generatorOn = other.generatorOn;
    _worker = other._worker;
    return *this;
}

/// @brief Reset the engine to its initial state
void EngineGen::reset()
{
    _worker.init();
    clearGravity();
    camera.resetPosition();
    generatorOn = true;
    simulationOn = false;
}

/// @brief Set the uniform for the shader program and set it to be used
/// @param frameTime The current time for the frame
/// @param cursorX The mouse X coordinate
/// @param cursorY The mouse Y coordinate
/// @param height The window height
void EngineGen::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(_shader.program, "camera");
    
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, &toScreen.value[0][0]);
    _shader.setFloatUniform("maxTtl", _timeToLive);
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
void EngineGen::run()
{
    camera.move();
    _gravity[0].active = mousePressed;
    if (!simulationOn)
        return;

    _worker.Map();
    if (generatorOn)
        _worker.generate(_particlePerFrame);
    _worker.call(_gravity);
    _worker.Unmap();
}

/// @brief Increment the number of particle generated per frame 
void EngineGen::ppfUp()
{
    if (_particlePerFrame == MAX_PPF)
    {
        std::cout << "Particle per frame at max value : " << _particlePerFrame << "\n";
        return;
    }
    _particlePerFrame += PPF_STEP;
    if (_particlePerFrame >= MAX_PPF)
        _particlePerFrame = MAX_PPF;
    std::cout << "Particle per frame increased, new value : " << _particlePerFrame << "\n";
}

/// @brief Decrement the number of particle generated per frame 
void EngineGen::ppfDown()
{
    if (_particlePerFrame == MIN_PPF)
    {
        std::cout << "Particle per frame at min value : " << _particlePerFrame << "\n";
        return;
    }   
    _particlePerFrame -= PPF_STEP;
    if (_particlePerFrame <= MIN_PPF)
        _particlePerFrame = MIN_PPF;
    std::cout << "Particle per frame decreased, new value : " << _particlePerFrame << "\n";
}
