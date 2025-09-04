#include "Window.hpp"

#include <iostream>
#include <chrono>

#include "Engine/AEngine.hpp"
#include "Engine/EngineGen.hpp"
#include "Engine/EngineStatic.hpp"
#include "FPSCounter.hpp"

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

bool Window::WasCreated() const
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
        engine->setGravity(winInstance->cursorX, winInstance->cursorY, winInstance->currentWidth, winInstance->currentHeight);
            
    else if (key == GLFW_KEY_H && action == GLFW_PRESS)
        engine->gravityOn = false;

    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
        engine->GravityUp();
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
        engine->GravityDown();

    else if (engine->initType == ENGINE_INIT_STATIC)
    {
        EngineStatic *engineS = reinterpret_cast<EngineGen *>(engine);
        if (key == GLFW_KEY_T && action == GLFW_PRESS)
            engineS->resetCube();
    }

    else if (engine->initType == ENGINE_INIT_GEN)
    {
        EngineGen *engineG = reinterpret_cast<EngineGen *>(engine);
        if (key == GLFW_KEY_T && action == GLFW_PRESS)
            engineG->generatorOn = !engineG->generatorOn ;
    }

}

void mouseCallback(GLFWwindow* window, int key, int action, int mods)
{
    (void)mods;

    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    AEngine *engine = winInstance->engine;
    

    if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
        engine->mousePressed = true;
    else if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE)
    {
        engine->mousePressed = false;
        engine->gravityOn = false;
    }
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
    glfwSetMouseButtonCallback(_window, mouseCallback);
    return 0;
}

void Window::RenderLoop()
{
    float currentFrame = 0.0f;
    FPSCounter counter(60, glfwGetTime());

    while(!glfwWindowShouldClose(_window))
    {
        currentFrame = glfwGetTime();
        glfwGetCursorPos(_window, &cursorX, &cursorY);
        counter.addFrame(currentFrame);

        if (counter.getFrame() % 30)
            glfwSetWindowTitle(_window, std::to_string(counter.getFPS()).c_str());

        if (engine->mousePressed)
            engine->setGravity(cursorX, cursorY, currentWidth, currentHeight);
        engine->run();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        engine->useShader(currentFrame, cursorX, cursorY, currentHeight);
        glBindVertexArray(engine->VAO);
        engine->draw();
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}