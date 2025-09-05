#include <iostream>
#include <random>

#include "math/transform.hpp"
#include "math/vec4.hpp"
#include "Engine/AEngine.hpp"


AEngine::AEngine(int particleQuantity)
{
    gravityOn = false;
    _gravityPos = vec3(0.0f, 0.0f, 0.0f);
    _particleQty = particleQuantity;
    simulationOn = false;
    mousePressed = false;
    gravityStrength = BASE_GRAVITY;

}

AEngine::AEngine(const AEngine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    _shader = other._shader;
    camera = other.camera;
    _particleQty = other._particleQty;
    simulationOn = other.simulationOn;
    gravityOn = other.gravityOn;
    _gravityPos = other._gravityPos;
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
    _shader = other._shader;
    camera = other.camera;
    _particleQty = other._particleQty;
    simulationOn = other.simulationOn;
    gravityOn = other.gravityOn;
    _gravityPos = other._gravityPos;
    return *this;
}

void AEngine::deleteArrays()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void AEngine::draw() const
{
    glDrawArrays(GL_POINTS, 0, _particleQty);
}

vec3 AEngine::_cursorToWorld(float cursorX, float cursorY, float width, float height) const
{
    float cursorXNdc = 2 * cursorX / width - 1;
    float cursorYNdc = 2 * cursorY / height - 1;
    float depthNdc =
        (camera.far + camera.near - (2.0 * camera.near * camera.far) / _mouseDepth) 
        / (camera.far - camera.near);

    vec4 mouseNdc = vec4(cursorXNdc, -cursorYNdc, depthNdc, 1.0f);
    vec4 mouseWorld = inverse(camera.coordToScreenMatrix()) * mouseNdc;
    return mouseWorld * (1 / mouseWorld.w);
}

void AEngine::addGravity(float cursorX, float cursorY, float width, float height)
{
    _gravity.push_back(Gravity(_cursorToWorld(cursorX, cursorY, width, height)));
}

void AEngine::setMouseGravity(float cursorX, float cursorY, float width, float height)
{
    _gravity[0].SetPos(_cursorToWorld(cursorX, cursorY, width, height));
    _gravity[0].active = true;
}

void AEngine::clearGravity()
{
    _gravity.erase(++_gravity.begin(), _gravity.end());
}