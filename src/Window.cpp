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

    window = glfwCreateWindow(BASE_WIN_WIDTH, BASE_WIN_HEIGHT, "Particle System", NULL, NULL);
    glfwSetWindowUserPointer(window, this);
    engine = 0;
}

Window::Window(const Window& other)
{
    window = other.window;
    engine = other.engine;
}

Window::~Window()
{
}

Window &Window::operator=(const Window &other)
{
    if (this ==  &other)
        return *this;

    window = other.window;
    engine = other.engine;
    return *this;
}

/// @brief Bind the engine to the window
/// @param newEngine The engien to bind
void Window::bindEngine(AEngine *newEngine)
{
    glfwGetFramebufferSize(window, &currentWidth, &currentHeight);
    engine = newEngine;
    engine->camera.computeProjectionMatrix(currentHeight, currentWidth);
}

/// @brief Check if the window was succesfully created 
/// @return True if the window was sucessfully created, false if not
bool Window::WasCreated() const
{
    return window != NULL;
}

/// @brief Callback for the window size change events
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    (void)window;
    glViewport(0, 0, width, height);
    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    winInstance->currentHeight = height;
    winInstance->currentWidth = width;
    winInstance->engine->camera.computeProjectionMatrix(height, width);
}  

/// @brief Callback for the keyboard events
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
        engine->addGravity(winInstance->cursorX, winInstance->cursorY, winInstance->currentWidth, winInstance->currentHeight);
            
    else if (key == GLFW_KEY_H && action == GLFW_PRESS)
        engine->clearGravity();

    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
        engine->allGravityUp();
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
        engine->allGravityDown();

    else if (engine->initType == ENGINE_INIT_STATIC)
    {
        EngineStatic *engineS = reinterpret_cast<EngineStatic *>(engine);
        if (key == GLFW_KEY_T && action == GLFW_PRESS)
            engineS->resetCube();
    }

    else if (engine->initType == ENGINE_INIT_GEN)
    {
        EngineGen *engineG = reinterpret_cast<EngineGen *>(engine);
        if (key == GLFW_KEY_T && action == GLFW_PRESS)
            engineG->generatorOn = !engineG->generatorOn;
            
        else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
            engineG->ppfDown();
        else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
            engineG->ppfUp();
    }

}

/// @brief Callback for the nouse events
void mouseCallback(GLFWwindow* window, int key, int action, int mods)
{
    (void)mods;

    Window *winInstance = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    AEngine *engine = winInstance->engine;
    

    if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
        engine->mousePressed = true;
    else if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_RELEASE)
        engine->mousePressed = false;
}

/// @brief Initialize the glfw window and glad
/// @note Glad is initilsez here because it needs a context, in this case the window, to be initilazed
/// @return 0 in case of success, -1 if an error occurred
int Window::Init()
{
    glfwMakeContextCurrent(window);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
        return -1;

    glEnable(GL_DEPTH_TEST);
    currentWidth = BASE_WIN_WIDTH;
    currentHeight = BASE_WIN_HEIGHT;
    glViewport(0, 0, currentWidth, currentHeight);
    glfwGetCursorPos(window, &cursorX, &cursorY);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseCallback);
    return 0;
}

/// @brief Main render loop for the window 
void Window::RenderLoop()
{
    float currentFrame = 0.0f;
    FPSCounter counter(60, glfwGetTime());

    while(!glfwWindowShouldClose(window))
    {
        currentFrame = glfwGetTime();
        glfwGetCursorPos(window, &cursorX, &cursorY);
        counter.addFrame(currentFrame);

        if (counter.getFrame() == 0)
            glfwSetWindowTitle(window, std::to_string(counter.getFPS()).c_str());

        if (engine->mousePressed)
            engine->setMouseGravity(cursorX, cursorY, currentWidth, currentHeight);
            
        engine->run();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        engine->useShader(currentFrame, cursorX, cursorY, currentHeight);
        glBindVertexArray(engine->VAO);
        engine->draw();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}