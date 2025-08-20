#include "Window.hpp"

#include <iostream>
#include <format>

Window::Window()
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(baseWidth, baseHeight, "Particle System", NULL, NULL);
    glfwSetWindowUserPointer(_window, this);
    engine = 0;
}

Window::Window(Window& other)
{
    _window = other._window;
    engine = other.engine;
}

Window::~Window()
{
}

void Window::bindEngine(Engine *newEngine)
{
    glfwGetFramebufferSize(_window, &currentWidth, &currentHeight);
    engine = newEngine;
    engine->camera.computeProjectionMatrix(currentHeight, currentWidth);
}

bool Window::WasCreated()
{
    return _window != NULL;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    (void)window;
    glViewport(0, 0, width, height);
    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    winInstance->currentHeight = height;
    winInstance->currentWidth = width;
    winInstance->engine->camera.computeProjectionMatrix(height, width);
}  

int Window::Init()
{
    glfwMakeContextCurrent(_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        return -1;

    glViewport(0, 0, baseWidth, baseHeight);
    currentWidth = baseWidth;
    currentHeight = baseHeight;
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    glEnable(GL_DEPTH_TEST);
    return 0;
}

void Window::ProcessInput()
{
    if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(_window, true);

    if (glfwGetKey(_window, GLFW_KEY_Q) == GLFW_PRESS)
        engine->camera.direction.y += 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_E) == GLFW_PRESS)
        engine->camera.direction.y -= 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
        engine->camera.position.z -= 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
        engine->camera.position.x -= 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
        engine->camera.position.z += 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
        engine->camera.position.x += 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS)
        engine->camera.position.y += 0.1f;
    if (glfwGetKey(_window, GLFW_KEY_X) == GLFW_PRESS)
        engine->camera.position.y -= 0.1f;
}

void Window::RenderLoop()
{
    float currentFrame = 0.0f;
    float lastFrame = glfwGetTime();
    int fps = 0;
    // std::string title;

    while(!glfwWindowShouldClose(_window))
    {
        currentFrame = glfwGetTime();
        glfwGetCursorPos(_window, &cursorX, &cursorY);
        fps++;
        if (currentFrame - lastFrame > 0.5)
        {
            fps = fps / (currentFrame - lastFrame);
            glfwSetWindowTitle(_window, std::to_string(fps).c_str());
            lastFrame = currentFrame;
            fps = 0;
        }
        this->ProcessInput();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        engine->useShader(currentFrame, cursorX, cursorY, currentHeight);
        glBindVertexArray(engine->VAO);
        engine->draw();
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}