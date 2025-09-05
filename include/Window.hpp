#pragma once

#include "glad/gl.h"
#include <GLFW/glfw3.h>

#define BASE_WIN_HEIGHT 768
#define BASE_WIN_WDITH 1024

class AEngine;

/// @brief The class used to manage a glfw window
class Window
{
    private:
        /// @brief The glfw window pointer
        GLFWwindow *_window;
        /// @brief The current window height
        int _currentHeight;
        /// @brief The current window width
        int _currentWidth;
        /// @brief The current position of the cursor on the x axis
        double _cursorX;
        /// @brief The current position of the cursor on the y axis
        double _cursorY;
        /// @brief The engine used for the display inside the window
        AEngine *_engine;

    public:
        Window();
        Window(const Window &other);
        ~Window();

        Window &operator=(const Window &other);

        bool WasCreated() const;
        void bindEngine(AEngine *engine);
        int Init();
        void RenderLoop();
};