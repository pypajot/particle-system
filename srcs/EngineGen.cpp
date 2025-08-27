#include <random>

#include "EngineGen.hpp"

EngineGen::EngineGen(int particleQuantity) : AEngine(particleQuantity)
{
    initType = "generator";    
    particlePerFrame = particleQty / timeToLive;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBufferData(GL_ARRAY_BUFFER, particleQty * 7 * sizeof(float), 0, GL_STREAM_DRAW);

    reset();
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

void EngineGen::reset()
{
    camera.resetPosition();
    generatorOn = true;
    float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    currentParticle = 0;
    for (int i = 0; i < particleQty; i++)
        buffer[i * 7 + 6] = timeToLive;
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

void EngineGen::run()
{
    if (!generatorOn)
        return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disDistance(0.0f, 0.2f);
    std::uniform_real_distribution<> disY(0, 0.1f);
    std::uniform_real_distribution<> disXZ(0, M_PI * 2);
    std::uniform_real_distribution<> speedDis(2.0, 2.5f);
    float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    int index;
    float angleY;
    float angleXZ;
    float distance;
    int speed;

    for (int i = 0; i < particlePerFrame; i++)
    {
        index = (currentParticle + i) % particleQty;
        angleY = M_PI_2 - disY(gen);
        angleXZ = disXZ(gen);
        distance = disDistance(gen);
        speed = speedDis(gen);
        buffer[index * 7] = cos(angleY) * cos(angleXZ) * distance;
        buffer[index * 7 + 1] = sin(angleY) * distance;
        buffer[index * 7 + 2] = cos(angleY) * sin(angleXZ) * distance;
        buffer[index * 7 + 3] = cos(angleY) * cos(angleXZ) * speed;
        buffer[index * 7 + 4] = sin(angleY) * speed;
        buffer[index * 7 + 5] = cos(angleY) * sin(angleXZ) * speed;
        buffer[index * 7 + 6] = 0;
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    currentParticle = currentParticle + particlePerFrame % particleQty;
}