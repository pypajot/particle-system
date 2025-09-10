#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <iostream>

#include "Window.hpp"
#include "Engine/AEngine.hpp"
#include "Engine/EngineStatic.hpp"
#include "Engine/EngineGen.hpp"


uint getParticleQty(int ac, char **av)
{
    if (ac < 2)
        return 0;

    return atoi(av[1]);
}

int getGeneratorOption(int ac, char **av)
{
    if (ac < 3)
        return 0;

    std::string option(av[2]);
    if (option != "-g")
        return -1;

    if (ac < 4)
        return BASE_TTL;

    int ttl = atoi(av[3]);
    return ttl < BASE_TTL ? BASE_TTL : ttl;
}

int main(int ac, char **av)
{
    uint particleQty = getParticleQty(ac, av);
    uint ttl = getGeneratorOption(ac, av);

    if (particleQty == 0 || ttl < 0)
    {
        std::cerr << "Valid argument needed: format './particle <number of particle> [-g <time to live>]'\n";
        return 1;
    }

    glfwInit();

    Window window;
    
    if (!window.WasCreated())
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    if (window.Init())
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    AEngine *particle;
    if (ttl > 0)
        particle = new EngineGen(particleQty, ttl);
    else
        particle = new EngineStatic(particleQty);

        
    window.bindEngine(particle);

    std::cout << "Particle system started !\n\n";
    std::cout << "Controls:\n";
    std::cout << "WASD: move the camera\n";
    std::cout << "Space and X: move the camera up and down\n";
    std::cout << "A and E : rotate the camera\n";
    std::cout << "Enter: start the simulation\n";
    std::cout << "R: reset the simulation to its starting position\n";
    if (ttl > 0)
        std::cout << "T: Start/stop the generator\n";
    else
        std::cout << "T: Reset the simulation to a cubic position\n";
    std::cout << "G: add a gravity point at the cursor\n";
    std::cout << "H: clear the gravity points\n";
    std::cout << "You can also use the mouse to add a temporary gravity point\n";
    std::cout << "Up and down arrows: increase or decrease the strength of the active gravity points\n";
    if (ttl > 0)
        std::cout << "Right and left arrows: increase or decrease the number of particle generated per frame\n";

    window.RenderLoop();
    particle->deleteArrays();
    delete(particle);
    glfwTerminate();
    return 0;
}