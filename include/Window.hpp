#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#define BASE_WIN_HEIGHT 1080
#define BASE_WIN_WIDTH 1920

class AEngine;

/// @brief The class used to manage a glfw window
class Window
{
    public:
        /// @brief The glfw window pointer
        GLFWwindow *window;
        /// @brief The current window height
        int currentHeight;
        /// @brief The current window width
        int currentWidth;
        /// @brief The current position of the cursor on the x axis
        double cursorX;
        /// @brief The current position of the cursor on the y axis
        double cursorY;
        /// @brief The engine used for the display inside the window
        AEngine *engine;

        Window();
        Window(const Window &other);
        ~Window();

        Window &operator=(const Window &other);

        bool WasCreated() const;
        void bindEngine(AEngine *engine);
        int Init();
        void RenderLoop();
};