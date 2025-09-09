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

uint getGeneratorOption(int ac, char **av)
{
    if (ac < 3)
        return 0;

    std::string option(av[2]);
    if (option != "-g")
        return 0;

    if (ac < 4)
        return BASE_TTL;

    uint ttl = atoi(av[3]);
    return ttl < BASE_TTL ? BASE_TTL : ttl;
}

int main(int ac, char **av)
{
    uint particleQty = getParticleQty(ac, av);
    uint ttl = getGeneratorOption(ac, av);

    if (particleQty == 0)
    {
        std::cerr << "Valid argument needed: format './particle <number>'\n";
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
        particle = new EngineGen(particleQty);
    else
        particle = new EngineStatic(particleQty);

    window.bindEngine(particle);
    window.RenderLoop();
    particle->deleteArrays();
    delete(particle);
    glfwTerminate();
    return 0;
}