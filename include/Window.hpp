#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

class AEngine;

class Window
{
    private:
        GLFWwindow *_window;

        const int _baseHeight = 768;
        const int _baseWidth = 1024;
        int _currentHeight;
        int _currentWidth;
        double _cursorX;
        double _cursorY;
        AEngine *_engine;

    public:
        Window();
        Window(Window &other);
        ~Window();

        bool WasCreated() const;
        void bindEngine(AEngine *engine);
        int Init();
        void RenderLoop();
};