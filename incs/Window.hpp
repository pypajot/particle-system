#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Window
{
    private:
        GLFWwindow *_window;

    public:
        const int baseHeight = 600;
        const int baseWidth = 800;

        Window();
        Window(Window &other);
        ~Window();

        bool WasCreated();
        int Init();
        void ProcessInput();
        void RenderLoop();

};