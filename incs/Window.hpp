#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Engine.hpp"

class Window
{
    private:
        GLFWwindow *_window;

    public:
        const int baseHeight = 600;
        const int baseWidth = 800;
        int currentHeight;
        int currentWidth;
        double cursorX;
        double cursorY;
        Engine *engine;


        Window();
        Window(Window &other);
        ~Window();

        bool WasCreated();
        void bindEngine(Engine *engine);
        int Init();
        void ProcessInput();
        void RenderLoop();

};