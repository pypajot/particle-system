#include <iostream>
#include <random>

#include "math/transform.hpp"
#include "Engine/AEngine.hpp"


AEngine::AEngine(int particleQuantity)
{
    gravityOn = false;
    gravityPos = vec3(0.0f, 0.0f, 0.0f);
    particleQty = particleQuantity;
    simulationOn = false;
    mousePressed = false;
    gravityStrength = BASE_GRAVITY;

}

AEngine::AEngine(const AEngine &other)
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

AEngine &AEngine::operator=(const AEngine &other)
{
    if (this == &other)
        return *this;
        
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

void AEngine::draw()
{
    glDrawArrays(GL_POINTS, 0, particleQty);
}

void AEngine::setGravity(float cursorX, float cursorY, float width, float height)
{
    float cursorXNdc = 2 * cursorX / currentWidth - 1
    float cursorYNdc = 2 * cursorY / currentHeight - 1
    float depthNdc =
        (camera.far + camera.near - (2.0 * camera.near * camera.far) / mouseDepth) 
        / (camera.far - camera.near);

    vec4 mouseNdc = vec4(cursorXNdc, -cursorYNdc, depthNdc, 1.0f);
    vec4 mouseWorld = inverse(camera.coordToScreenMatrix()) * mouseNdc;
    gravityPos = mouseWorld / mouseWorld.w;
    gravityOn = true;
}

void AEngine::GravityUp()
{
    if (gravityStrength >= MAX_GRAVITY)
        return;
    gravityStrength += 0.1f;
}

void AEngine::GravityDown()
{
    if (gravityStrength <= MIN_GRAVITY)
        return;
    gravityStrength -= 0.1f;
}