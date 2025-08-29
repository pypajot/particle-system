#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include "AEngine.hpp"

class Window
{
    private:
        GLFWwindow *_window;

    public:
        const int baseHeight = 768;
        const int baseWidth = 1024;
        int currentHeight;
        int currentWidth;
        double cursorX;
        double cursorY;
        AEngine *engine;


        Window();
        Window(Window &other);
        ~Window();

        bool WasCreated();
        void bindEngine(AEngine *engine);
        int Init();
        void ProcessInput();
        void RenderLoop();

};