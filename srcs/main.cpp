#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Window.hpp"
#include "Engine.hpp"


int parseArgs(int ac, char **av)
{
    if (ac != 2)
        return 0;
    return atoi(av[1]);
}

int main(int ac, char **av)
{
    int particleQty = parseArgs(ac, av);
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

    Engine particle(particleQty);
    window.bindEngine(&particle);
    window.RenderLoop();

    glfwTerminate();
    return 0;
}