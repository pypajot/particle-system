#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Engine.hpp"

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