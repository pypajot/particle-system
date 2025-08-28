#include "gl.h"
#include <GLFW/glfw3.h>

#include <iostream>

#include "Window.hpp"
#include "AEngine.hpp"
#include "EngineStatic.hpp"
#include "EngineGen.hpp"


int getParticleQty(int ac, char **av)
{
    if (ac == 1 || ac > 3)
        return 0;

    return atoi(av[1]);
}

bool getGeneratorOption(int ac, char **av)
{
    if (ac == 2)
        return false;

    std::string option(av[2]);
    if (option == "-g")
        return true;

    return false;
}

int main(int ac, char **av)
{
    int particleQty = getParticleQty(ac, av);
    bool hasGenerator = getGeneratorOption(ac, av);

    if (particleQty == 0)
        return 1;

    glfwInit();

    Window window;
    
    if (!window.WasCreated())
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    if (window.Init())
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }
    AEngine *particle;
    if (hasGenerator)
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