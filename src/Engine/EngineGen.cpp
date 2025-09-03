#include <random>
#include <cmath>
#include "EngineGen.hpp"


EngineGen::EngineGen(int particleQuantity) : AEngine(particleQuantity)
{
    vertexPath = "shaders/vertexShaderGen.vs";
    Shader s(vertexPath.c_str(), fragmentPath.c_str());
    shader = s;
    initType = "generator";    
    particlePerFrame = particleQty / timeToLive;
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
    CudaWorker test(VBO, particleQty);
    gravity = test;
    reset();
}

void EngineGen::reset()
{
    camera.resetPosition();
    generatorOn = true;
    gravity.initGen(timeToLive);
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

void EngineGen::run()
{
    // if (generatorOn)
    // {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<> disY(0, 0.2f);
    //     std::uniform_real_distribution<> disXZ(0, M_PI * 2);
    //     std::uniform_real_distribution<> speedDis(0.5f, 1.0f);
    //     float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    //     int index;
    //     float angleY;
    //     float angleXZ;
    //     float speed;
    
    //     for (int i = 0; i < particlePerFrame; i++)
    //     {
    //         index = (currentParticle + i) % particleQty;
    //         angleY = M_PI_2 - disY(gen);
    //         angleXZ = disXZ(gen);
    //         speed = speedDis(gen);
    //         buffer[index * 7] = 0.0f;
    //         buffer[index * 7 + 1] = -0.5f;
    //         buffer[index * 7 + 2] = 0.0f;
    //         buffer[index * 7 + 3] = cos(angleY) * cos(angleXZ) * speed;
    //         buffer[index * 7 + 4] = sin(angleY) * speed;
    //         buffer[index * 7 + 5] = cos(angleY) * sin(angleXZ) * speed;
    //         buffer[index * 7 + 6] = 0.0f;
    //     }
    //     glUnmapBuffer(GL_ARRAY_BUFFER);
    //     currentParticle = currentParticle + particlePerFrame % particleQty;
    // }

    gravity.callGen(gravityPos, gravityOn);
}