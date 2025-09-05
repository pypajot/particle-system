#include "glad/gl.h"
#include <GLFW/glfw3.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <ios>

#include "Shader.hpp"

Shader::Shader()
{
    program = 0;
}

/// @brief Load a shader given its path
/// @param shaderPath THe path to the shader
/// @return The shader code
std::string loadShader(std::string shaderPath)
{
    std::ifstream shaderFile(shaderPath);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    shaderFile.close();
    return shaderStream.str();
}

/// @brief Compile the shader passed as argument
/// @param shaderSourceString  The code of the shader to compile
/// @param shaderType The type of shader to compile 
/// @return The int representing the shader 
int compileShader(std::string shaderSourceString, int shaderType)
{
    const char *shaderSource = shaderSourceString.c_str();
    unsigned int shader;
    shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);
    return shader;
}

/// @brief Constructs a shader program using the vertex and fragment shaders passed as arguments
/// @param vertexPath The path to the vertex shader
/// @param fragmentPath The path to the fragment shader
Shader::Shader(const char *vertexPath, const char *fragmentPath)
{
    std::string vertexShaderCode = loadShader(vertexPath);
    std::string fragmentShaderCode = loadShader(fragmentPath);
    int  success;
    char infoLog[512];

    int vertexShader = compileShader(vertexShaderCode, GL_VERTEX_SHADER);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Error during vertex shader conpilation\n" << infoLog << std::endl;
    }

    int fragmentShader = compileShader(fragmentShaderCode, GL_FRAGMENT_SHADER);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Error during fragment shader compilation\n" << infoLog << std::endl;
    }

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if(!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Error during shader program linking\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    program = shaderProgram;
}

Shader::Shader(const Shader &other)
{
    program = other.program;
}

Shader Shader::operator=(const Shader &other)
{
    program = other.program;
    return *this;
}

Shader::~Shader()
{
}

/// @brief Set a floating point value to be used by the shader program
/// @param name The name a the value
/// @param value The value
void Shader::setFloatUniform(const char *name, float value) const
{
    int loc = glGetUniformLocation(program, name);
    if (loc == -1)
        std::cerr << "Error setting uniform: " << name << "\n";
    glUniform1f(loc, value);
}

/// @brief Set a integer value to be used by the shader program
/// @param name The name a the value
/// @param value The value
void Shader::setIntUniform(const char *name, int value) const
{
    int loc = glGetUniformLocation(program, name);
    if (loc == -1)
        std::cerr << "Error setting uniform: " << name << "\n";
    glUniform1i(loc, value);
}

/// @brief Wrapper around the glUseProgram function for the current shader program
void Shader::use() const
{
    glUseProgram(program);
}
