#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>

#include "AEngine.hpp"

AEngine::AEngine(int particleQuantity)
{
    Shader s(vertexPath.c_str(), fragmentPath.c_str());
    shader = s;
    gravityOn = false;
    gravityPos = glm::vec3(0.0f, 0.0f, 0.0f);
    particleQty = particleQuantity;
    simulationOn = false;
}

AEngine::AEngine(AEngine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    shader = other.shader;
    camera = other.camera;
    particleQty = other.particleQty;
    simulationOn = other.simulationOn;
    gravityOn = other.gravityOn;
    gravityPos = other.gravityPos;
}

AEngine::~AEngine()
{
}

AEngine &AEngine::operator=(AEngine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    shader = other.shader;
    camera = other.camera;
    particleQty = other.particleQty;
    simulationOn = other.simulationOn;
    gravityOn = other.gravityOn;
    gravityPos = other.gravityPos;
    return *this;
}

void AEngine::deleteArrays()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void AEngine::useShader(float frameTime, float cursorX, float cursorY, float height)
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

void AEngine::draw()
{
    glDrawArrays(GL_POINTS, 0, particleQty);
}

void AEngine::setGravity(float cursorX, float cursorY)
{
    glm::vec4 test = glm::vec4(cursorX, cursorY, 0.9f, 1.0f);
    glm::mat4 screenToCam = glm::inverse(camera.coordToScreenMatrix());
    glm::vec4 testResult = screenToCam * test;
    float inv = 1 / testResult.w;
    testResult *= inv;
    gravityPos = testResult;
}

// void AEngine::run()
// {
// }
