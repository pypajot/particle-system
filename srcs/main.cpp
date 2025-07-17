#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Window.hpp"


int main()
{
    
    glfwInit();
    
    
    Window window;
    
    if (!window.WasCreated())
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    if (!window.Init())
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    window.RenderLoop();

    glfwTerminate();
    return 0;
}