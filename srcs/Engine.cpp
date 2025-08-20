#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>

#include "Engine.hpp"

// Engine::Engine()
// {
    
// }



void initBuffer(int particleQty)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disZ(-M_PI_2, M_PI_2);
    std::uniform_real_distribution<> disXY(0, M_PI * 2);
    std::uniform_real_distribution<> speedDis(0, 0.1f);
    float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    float angleZ;
    float angleXY;
    for (int i = 0; i < particleQty; i++)
    {
        angleZ = disZ(gen);
        angleXY = disXY(gen);
        buffer[i * 6] = cos(angleZ) * cos(angleXY);
        buffer[i * 6 + 1] = cos(angleZ) * sin(angleXY);
        buffer[i * 6 + 2] = sin(angleZ);
        buffer[i * 6 + 3] = speedDis(gen);
        buffer[i * 6 + 4] = speedDis(gen);
        buffer[i * 6 + 5] = speedDis(gen);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

Engine::Engine(int particleQuantity)
{
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBufferData(GL_ARRAY_BUFFER, particleQuantity * 6 * sizeof(float), 0, GL_STREAM_DRAW);
    initBuffer(particleQuantity);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    Shader s(vertexPath.c_str(), fragmentPath.c_str());
    shader = s;
    particleQty = particleQuantity;
}

Engine::Engine(Engine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    shader = other.shader;
    camera = other.camera;
    particleQty = other.particleQty;
}

Engine::~Engine()
{
}

Engine Engine::operator=(Engine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    shader = other.shader;
    camera = other.camera;
    particleQty = other.particleQty;
    return *this;
}

void Engine::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    glm::mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(shader.program, "camera");
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, glm::value_ptr(toScreen));
    shader.setFloatUniform("frameTimeX", (1 + sin(frameTime)) / 2);
    shader.setFloatUniform("frameTimeY", (1 + sin(frameTime + 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("frameTimeZ", (1 + sin(frameTime - 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("cursorX", cursorX);
    shader.setFloatUniform("cursorY", cursorY);
    shader.setFloatUniform("height", height);
    shader.use();

}

void Engine::draw()
{
    glDrawArrays(GL_POINTS, 0, particleQty);
}