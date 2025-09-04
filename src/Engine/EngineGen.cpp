#include <random>
#include <cmath>

#include "Engine/EngineGen.hpp"


EngineGen::EngineGen() : AEngine()
{
}

EngineGen::EngineGen(int particleQuantity) : AEngine(particleQuantity)
{
    initType = ENGINE_INIT_GEN;    
    vertexPath = "shaders/vertexShaderGen.vs";
    shader = Shader(vertexPath.c_str(), fragmentPath.c_str());

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBufferData(GL_ARRAY_BUFFER, particleQty * 7 * sizeof(float), 0, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    worker = std::unique_ptr<AWorker>(new WorkerGen(VBO, particleQty, timeToLive, particlePerFrame));
    reset();
}

EngineGen::EngineGen(const EngineGen &other) : AEngine(other)
{
    currentParticle = other.currentParticle;
    generatorOn = other.generatorOn;
    worker = other.worker;
}

void EngineGen::reset()
{
    camera.resetPosition();
    generatorOn = true;
    worker->initGen(timeToLive);
    simulationOn = false;
}

void EngineGen::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(shader.program, "camera");
    
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, &toScreen.value[0][0]);
    shader.setFloatUniform("maxTtl", timeToLive);
    shader.setFloatUniform("frameTimeX", (1 + sin(frameTime)) / 2);
    shader.setFloatUniform("frameTimeY", (1 + sin(frameTime + 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("frameTimeZ", (1 + sin(frameTime - 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("cursorX", cursorX);
    shader.setFloatUniform("cursorY", cursorY);
    shader.setFloatUniform("height", height);
    shader.setFloatUniform("near", camera.near);
    shader.setFloatUniform("far", camera.far);
    shader.setFloatUniform("mouseDepth", mouseDepth);
    shader.use();
}
