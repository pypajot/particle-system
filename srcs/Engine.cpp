#include "Engine.hpp"

#include <fstream>

Engine::Engine()
{
    glGenBuffers(1, VBO);
}

Engine::Engine(Engine &other)
{
    VBO[0] = other.VBO[0];
}

Engine::~Engine()
{
}

void Engine::Init()
{
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);  
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

}

int Engine::loadShader()
{
    std::string shader;

    std::ifstream shaderFile("")
}

