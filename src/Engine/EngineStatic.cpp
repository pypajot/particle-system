#include <random>

#include "Engine/EngineStatic.hpp"

#include <iostream>

void EngineStatic::initCube()
{
    float size = 0.7f;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dis(-size, size);
    std::uniform_int_distribution<> disSide(1, 6);
    std::uniform_real_distribution<> speedDis(0, 0.1f);

    float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    int side;
    for (int i = 0; i < particleQty; i++)
    {
        side = disSide(gen);
        if (side == 1)
        {
            buffer[i * 6] = -size;
            buffer[i * 6 + 1] = dis(gen);
            buffer[i * 6 + 2] = dis(gen); 
        }
        else if (side == 2)
        {
            buffer[i * 6] = size;
            buffer[i * 6 + 1] = dis(gen);
            buffer[i * 6 + 2] = dis(gen); 
        }
        else if (side == 3)
        {
            buffer[i * 6] = dis(gen);
            buffer[i * 6 + 1] = -size;
            buffer[i * 6 + 2] = dis(gen); 
        }
        else if (side == 4)
        {
            buffer[i * 6] = dis(gen);
            buffer[i * 6 + 1] = size;
            buffer[i * 6 + 2] = dis(gen); 
        }
        else if (side == 5)
        {
            buffer[i * 6] = dis(gen);
            buffer[i * 6 + 1] = dis(gen);
            buffer[i * 6 + 2] = -size; 
        }
        else if (side == 6)
        {
            buffer[i * 6] = dis(gen);
            buffer[i * 6 + 1] = dis(gen);
            buffer[i * 6 + 2] = size; 
        }
        buffer[i * 6 + 3] = speedDis(gen);
        buffer[i * 6 + 4] = speedDis(gen);
        buffer[i * 6 + 5] = speedDis(gen);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    simulationOn = false;
}

void EngineStatic::initSphere()
{
    gravity.init();
}

void EngineStatic::reset()
{
    initSphere();
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

EngineStatic::EngineStatic(int particleQuantity) : AEngine(particleQuantity)
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

    gravity = WorkerStatic(VBO, particleQty);
    initSphere();
}
