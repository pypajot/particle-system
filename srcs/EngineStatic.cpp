#include <random>

#include "EngineStatic.hpp"

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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disY(-M_PI_2, M_PI_2);
    std::uniform_real_distribution<> disXZ(0, M_PI * 2);
    std::uniform_real_distribution<> speedDis(0, 0.1f);
    float *buffer = reinterpret_cast<float *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    float angleY;
    float angleXZ;
    for (int i = 0; i < particleQty; i++)
    {
        angleY = disY(gen);
        angleXZ = disXZ(gen);
        buffer[i * 6] = cos(angleY) * cos(angleXZ);
        buffer[i * 6 + 1] = sin(angleY);
        buffer[i * 6 + 2] = cos(angleY) * sin(angleXZ);
        buffer[i * 6 + 3] = speedDis(gen);
        buffer[i * 6 + 4] = speedDis(gen);
        buffer[i * 6 + 5] = speedDis(gen);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

void EngineStatic::reset()
{
    initSphere();
    camera.resetPosition();
    simulationOn = false;
}

EngineStatic::EngineStatic(int particleQuantity) : AEngine(particleQuantity)
{
    initType = "static";
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBufferData(GL_ARRAY_BUFFER, particleQty * 6 * sizeof(float), 0, GL_STREAM_DRAW);

    initSphere();
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void EngineStatic::run()
{
}