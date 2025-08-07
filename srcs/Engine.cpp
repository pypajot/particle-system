#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "Engine.hpp"

Engine::Engine()
{
    glGenBuffers(1, VBO);
    glGenVertexArrays(1, &VAO);
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);  
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    Shader s(vertexPath.c_str(), fragmentPath.c_str());
    shader = s;
}

Engine::Engine(Engine &other)
{
    VBO[0] = other.VBO[0];
    VAO = other.VAO;
    shader = other.shader;
    camera = other.camera;
}

Engine::~Engine()
{
}

void Engine::useShader(float height, float width)
{
    int camLoc = glGetUniformLocation(shader.program, "camera");
    shader.use();
    glm::mat4 camMatrix = camera.coordToScreenMatrix(height, width);
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, glm::value_ptr(camMatrix));
}