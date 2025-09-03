#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

class AEngine;

class Window
{
    private:
        GLFWwindow *_window;

        const int baseHeight = 768;
        const int baseWidth = 1024;
        int currentHeight;
        int currentWidth;
        double cursorX;
        double cursorY;
        AEngine *engine;

    public:
        Window();
        Window(Window &other);
        ~Window();

        bool WasCreated() const;
        void bindEngine(AEngine *engine);
        int Init();
        void RenderLoop();
};