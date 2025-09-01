#include <iostream>
#include <random>

#include "math.hpp"
#include "AEngine.hpp"


AEngine::AEngine(int particleQuantity)
{
    Shader s(vertexPath.c_str(), fragmentPath.c_str());
    shader = s;
    gravityOn = false;
    gravityPos = vec3(0.0f, 0.0f, 0.0f);
    particleQty = particleQuantity;
    simulationOn = false;
    mousePressed = false;
}

// AEngine::AEngine(AEngine &other)
// {
//     VBO = other.VBO;
//     VAO = other.VAO;
//     shader = other.shader;
//     camera = other.camera;
//     particleQty = other.particleQty;
//     simulationOn = other.simulationOn;
//     gravityOn = other.gravityOn;
//     gravityPos = other.gravityPos;
// }

AEngine::~AEngine()
{
}

// AEngine &AEngine::operator=(AEngine &other)
// {
//     VBO = other.VBO;
//     VAO = other.VAO;
//     shader = other.shader;
//     camera = other.camera;
//     particleQty = other.particleQty;
//     simulationOn = other.simulationOn;
//     gravityOn = other.gravityOn;
//     gravityPos = other.gravityPos;
//     return *this;
// }

void AEngine::deleteArrays()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void AEngine::useShader(float frameTime, float cursorX, float cursorY, float height)
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

void AEngine::draw()
{
    glDrawArrays(GL_POINTS, 0, particleQty);
}

void AEngine::setGravity(float cursorX, float cursorY)
{
    float depthNdc = (camera.far + camera.near - (2.0 * camera.near * camera.far) / mouseDepth) / (camera.far - camera.near);

    vec4 test = vec4(cursorX, -cursorY, depthNdc, 1.0f);
    mat4 screenToCam = inverse(camera.coordToScreenMatrix());
    vec4 testResult = screenToCam * test;
    testResult *= 1 / testResult.w;
    gravityPos = testResult;
}

