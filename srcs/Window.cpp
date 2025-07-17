#include "Window.hpp"

#include <iostream>

Window::Window()
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(baseWidth, baseHeight, "Particle System", NULL, NULL);
}

Window::Window(Window& other)
{
    _window = other._window;
}

Window::~Window()
{
}

bool Window::WasCreated()
{
    return _window != NULL;
}

void size_callback(GLFWwindow* window, int width, int height)
{
    (void)window;
    glViewport(0, 0, width, height);
}  

int Window::Init()
{
    glfwMakeContextCurrent(_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        return -1;

    glViewport(0, 0, baseWidth, baseHeight);
    glfwSetWindowSizeCallback(_window, size_callback);
    return 0;
}

void Window::ProcessInput()
{
    if(glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(_window, true);
}

void Window::RenderLoop()
{
    while(!glfwWindowShouldClose(_window))
    {
        this->ProcessInput();
        glfwSwapBuffers(_window);
        glfwPollEvents();    
    }
}