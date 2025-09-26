#include <iostream>
#include <random>

#include "math/transform.hpp"
#include "math/vec4.hpp"
#include "Engine/AEngine.hpp"
#include "constants.hpp"

AEngine::AEngine()
{
}

AEngine::AEngine(int particleQuantity)
{
    _gravity = std::vector<Gravity>(1);
    _particleQty = particleQuantity;
    simulationOn = false;
    mousePressed = false;

}

AEngine::AEngine(const AEngine &other)
{
    VBO = other.VBO;
    VAO = other.VAO;
    _shader = other._shader;
    camera = other.camera;
    _particleQty = other._particleQty;
    simulationOn = other.simulationOn;
    _gravity = other._gravity;
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
    _gravity = other._gravity;
    return *this;
}

/// @brief Delete the vertex and buffer arrays initialized with the init() method
void AEngine::deleteArrays()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

/// @brief Wrapper around the glDrawArrays function to draw points from the vertex array
void AEngine::draw() const
{
    glDrawArrays(GL_POINTS, 0, _particleQty);
}

/// @brief Compute the world coordinates of the cursor
/// @param cursorX The cursor X value
/// @param cursorY The cursor Y valuie
/// @param width The window width
/// @param height The window height
/// @return The world coordinate corresponding to the cursor
/// @note The cursor is assumed to be at a distance of _mouseDepth from the screen in world coordinates
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

/// @brief Add a gravity point to the world at the coordinates pointed to by the cursor
/// @param cursorX The cursor X value
/// @param cursorY The cursor Y value
/// @param width The window width
/// @param height The window height
void AEngine::addGravity(float cursorX, float cursorY, float width, float height)
{
    if (_gravity.size() == MAX_GRAVITY_POINTS)
    {
        std::cout << "Max gravity points reached, cannot add more.\n";
        return;
    }
    _gravity.push_back(Gravity(_cursorToWorld(cursorX, cursorY, width, height)));
}

/// @brief Update the gravity point corresponding to the mouse to match the cursor coordinates and acivate it
/// @param cursorX The cursor X value
/// @param cursorY The cursor Y value
/// @param width The window width
/// @param height The window height
void AEngine::setMouseGravity(float cursorX, float cursorY, float width, float height)
{
    _gravity[0].SetPos(_cursorToWorld(cursorX, cursorY, width, height));
    _gravity[0].active = true;
}

/// @brief Clear all gravity points
/// @note Only deactivate the point corresponding to the mouse 
void AEngine::clearGravity()
{
    _gravity.erase(_gravity.begin() + 1, _gravity.end());
    _gravity[0].active = false;
}

/// @brief Increase the strength of all active gravity points
void AEngine::allGravityUp()
{
    for (auto it = _gravity.begin(); it < _gravity.end(); it++)
        if (it->active)
            it->GravityUp();
}

/// @brief Decrease the strength of all active gravity points
void AEngine::allGravityDown()
{
    for (auto it = _gravity.begin(); it < _gravity.end(); it++)
        if (it->active)
            it->GravityDown();
}
