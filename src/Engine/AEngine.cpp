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

void AEngine::setGravity(float cursorX, float cursorY, float width, float height)
{
    float cursorXNdc = 2 * cursorX / width - 1;
    float cursorYNdc = 2 * cursorY / height - 1;
    float depthNdc =
        (camera.far + camera.near - (2.0 * camera.near * camera.far) / _mouseDepth) 
        / (camera.far - camera.near);

    vec4 mouseNdc = vec4(cursorXNdc, -cursorYNdc, depthNdc, 1.0f);
    vec4 mouseWorld = inverse(camera.coordToScreenMatrix()) * mouseNdc;
    _gravityPos = mouseWorld * (1 / mouseWorld.w);
    gravityOn = true;
}

void AEngine::GravityUp()
{
    if (gravityStrength >= MAX_GRAVITY)
    {
        std::cout << "Gravity strength at max value : " << gravityStrength << "\n";
        return;
    }
    gravityStrength += 0.1f;
    std::cout << "Gravity strength increased, new value : " << gravityStrength << "\n";
}

void AEngine::GravityDown()
{
    if (gravityStrength <= MIN_GRAVITY)
    {
        std::cout << "Gravity strength at min value : " << gravityStrength << "\n";
        return;
    }
    gravityStrength -= 0.1f;
    std::cout << "Gravity strength decreased, new value : " << gravityStrength << "\n";
}