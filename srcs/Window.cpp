#include "Window.hpp"

#include <iostream>

Window::Window()
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
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

void Window::bindEngine(AEngine *newEngine)
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

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    AEngine *engine = winInstance->engine;
    
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    else if (key == GLFW_KEY_Q && action == GLFW_PRESS)
        engine->camera.rotateLeftRight += 1;
    else if (key == GLFW_KEY_Q && action == GLFW_RELEASE)
        engine->camera.rotateLeftRight -= 1;
    else if (key == GLFW_KEY_E && action == GLFW_PRESS)
        engine->camera.rotateLeftRight -= 1;
    else if (key == GLFW_KEY_E && action == GLFW_RELEASE)
        engine->camera.rotateLeftRight += 1;

    else if (key == GLFW_KEY_W && action == GLFW_PRESS)
        engine->camera.moveFrontBack += 1;
    else if (key == GLFW_KEY_W && action == GLFW_RELEASE)
        engine->camera.moveFrontBack -= 1;
    else if (key == GLFW_KEY_S && action == GLFW_PRESS)
        engine->camera.moveFrontBack -= 1;
    else if (key == GLFW_KEY_S && action == GLFW_RELEASE)
        engine->camera.moveFrontBack += 1;

    else if (key == GLFW_KEY_A && action == GLFW_PRESS)
        engine->camera.moveLeftRight += 1;
    else if (key == GLFW_KEY_A && action == GLFW_RELEASE)
        engine->camera.moveLeftRight -= 1;
    else if (key == GLFW_KEY_D && action == GLFW_PRESS)
        engine->camera.moveLeftRight -= 1;
    else if (key == GLFW_KEY_D && action == GLFW_RELEASE)
        engine->camera.moveLeftRight += 1;

    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        engine->camera.moveUpDown += 1;
    else if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
        engine->camera.moveUpDown -= 1;
    else if (key == GLFW_KEY_X && action == GLFW_PRESS)
        engine->camera.moveUpDown -= 1;
    else if (key == GLFW_KEY_X && action == GLFW_RELEASE)
        engine->camera.moveUpDown += 1;

    else if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        engine->simulationOn = !engine->simulationOn;

    else if (key == GLFW_KEY_R && action == GLFW_PRESS)
        engine->reset();

    else if (key == GLFW_KEY_G && action == GLFW_PRESS)
        engine->setGravity(
            2 * winInstance->cursorX / winInstance->currentWidth - 1,
            -(2 * winInstance->cursorY / winInstance->currentHeight - 1)
        );
    else if (key == GLFW_KEY_H && action == GLFW_PRESS)
        engine->gravityOn = false;

}

int Window::Init()
{
    glfwMakeContextCurrent(_window);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
        return -1;

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, baseWidth, baseHeight);
    currentWidth = baseWidth;
    currentHeight = baseHeight;
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    glfwSetKeyCallback(_window, keyCallback);
    return 0;
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
        engine->camera.move();

        if (engine->simulationOn)
            engine->run();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        engine->useShader(currentFrame, cursorX, cursorY, currentHeight);
        glBindVertexArray(engine->VAO);
        engine->draw();
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}