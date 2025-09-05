#include "Window.hpp"

#include <iostream>
#include <chrono>

#include "Engine/AEngine.hpp"
#include "Engine/EngineGen.hpp"
#include "Engine/EngineStatic.hpp"
#include "FPSCounter.hpp"

/// @brief The constructor for the window class, using opengl 4.6 core
Window::Window()
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(_baseWidth, _baseHeight, "Particle System", NULL, NULL);
    glfwSetWindowUserPointer(_window, this);
    _engine = 0;
}

Window::Window(const Window& other)
{
    _window = other._window;
    _engine = other._engine;
}

Window::~Window()
{
}

Window &Window::operator=(const Window &other)
{
    if (this ==  &other)
        return *this;

    _window = other._window;
    _engine = other._engine;
    return *this;
}

/// @brief Bind the engine to the window
/// @param newEngine The engien to bind
void Window::bindEngine(AEngine *newEngine)
{
    glfwGetFramebufferSize(_window, &_currentWidth, &_currentHeight);
    _engine = newEngine;
    _engine->camera.computeProjectionMatrix(_currentHeight, _currentWidth);
}

/// @brief Check if the window was succesfully created 
/// @return True if the window was sucessfully created, false if not
bool Window::WasCreated() const
{
    return _window != NULL;
}

/// @brief Callback for the window size change events
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    (void)window;
    glViewport(0, 0, width, height);
    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    winInstance->_currentHeight = height;
    winInstance->_currentWidth = width;
    winInstance->_engine->camera.computeProjectionMatrix(height, width);
}  

/// @brief Callback for the keyboard events
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;

    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    AEngine *engine = winInstance->_engine;
    
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
        engine->setGravity(winInstance->_cursorX, winInstance->_cursorY, winInstance->_currentWidth, winInstance->_currentHeight);
            
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
            engineG->generatorOn = !engineG->generatorOn;
            
        else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
            engineG->ppfUp();
        else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
            engineG->ppfDown();
    }

}

/// @brief Callback for the nouse events
void mouseCallback(GLFWwindow* window, int key, int action, int mods)
{
    (void)mods;

    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    AEngine *engine = winInstance->_engine;
    

    if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
        engine->mousePressed = true;
    else if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE)
    {
        engine->mousePressed = false;
        engine->gravityOn = false;
    }
}

/// @brief Initialize the glfw window and glad
/// @note Glad is initilsez here because it needs a context, in this case the window, to be initilazed
/// @return 0 in case of success, -1 if an error occurred
int Window::Init()
{
    glfwMakeContextCurrent(_window);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
        return -1;

    glEnable(GL_DEPTH_TEST);
    _currentWidth = BASE_WIN_WDITH;
    _currentHeight = BASE_WIN_HEIGHT;
    glViewport(0, 0, _currentWidth, _currentHeight);
    glfwGetCursorPos(_window, &_cursorX, &_cursorY);
    glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    glfwSetKeyCallback(_window, keyCallback);
    glfwSetMouseButtonCallback(_window, mouseCallback);
    return 0;
}

/// @brief Main render loop for the window 
void Window::RenderLoop()
{
    float currentFrame = 0.0f;
    FPSCounter counter(60, glfwGetTime());

    while(!glfwWindowShouldClose(_window))
    {
        currentFrame = glfwGetTime();
        glfwGetCursorPos(_window, &_cursorX, &_cursorY);
        counter.addFrame(currentFrame);

        if (counter.getFrame() % 30)
            glfwSetWindowTitle(_window, std::to_string(counter.getFPS()).c_str());

        if (_engine->mousePressed)
            _engine->setGravity(_cursorX, _cursorY, _currentWidth, _currentHeight);
        _engine->run();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        _engine->useShader(currentFrame, _cursorX, _cursorY, _currentHeight);
        glBindVertexArray(_engine->VAO);
        _engine->draw();
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}