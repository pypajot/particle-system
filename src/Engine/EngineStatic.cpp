#include <random>

#include "Engine/EngineStatic.hpp"

#include <iostream>


EngineStatic::EngineStatic(int particleQuantity) : AEngine(particleQuantity), worker(VBO, particleQty)
{
    initType = ENGINE_INIT_STATIC;

    vertexPath = "shaders/vertexShader.vs";
    shader = Shader(vertexPath.c_str(), fragmentPath.c_str());;

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBufferData(GL_ARRAY_BUFFER, particleQty * 6 * sizeof(float), 0, GL_STREAM_DRAW);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    worker = std::unique_ptr<AWorker>(new WorkerStatic(VBO, particleQty));
    initSphere();
}

void EngineStatic::resetCube()
{
    worker->initCube();
    camera.resetPosition();
    simulationOn = false;
}

void EngineStatic::reset()
{
    worker->init();
    camera.resetPosition();
    simulationOn = false;
}

void EngineStatic::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(shader.program, "camera");
    
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, &toScreen.value[0][0]);
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

void EngineStatic::run()
{
    engine->camera.move();
    
    if (!simulationOn)
        return;
        
    worker->call(gravityPos, gravityOn, gravityStrength);
}