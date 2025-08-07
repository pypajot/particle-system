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

void Window::ProcessInput(Engine &engine)
{
    if(glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(_window, true);
    if(glfwGetKey(_window, GLFW_KEY_Q) == GLFW_PRESS)
        engine.camera.direction.y += 0.1;
    if(glfwGetKey(_window, GLFW_KEY_E) == GLFW_PRESS)
        engine.camera.direction.y -= 0.1;
    if(glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
        engine.camera.position.z -= 1;
    if(glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
        engine.camera.position.x -= 1;
    if(glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
        engine.camera.position.z += 1;
    if(glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
        engine.camera.position.x += 1;
}

void Window::RenderLoop(Engine &engine)
{
    while(!glfwWindowShouldClose(_window))
    {
        this->ProcessInput(engine);
        glClear(GL_COLOR_BUFFER_BIT);
        engine.useShader(baseHeight, baseWidth);
        glBindVertexArray(engine.VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}