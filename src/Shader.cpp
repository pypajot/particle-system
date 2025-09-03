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

std::string loadShader(std::string shaderPath)
{
    std::ifstream shaderFile(shaderPath);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    shaderFile.close();
    return shaderStream.str();
}

int compileShader(std::string shaderSourceString, int shaderType)
{
    const char *shaderSource = shaderSourceString.c_str();
    unsigned int shader;
    shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);
    return shader;
}

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
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    int fragmentShader = compileShader(fragmentShaderCode, GL_FRAGMENT_SHADER);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
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
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
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


void Shader::setFloatUniform(const char *name, float value) const
{
    int loc = glGetUniformLocation(program, name);
    if (loc == -1)
        std::cout << "Error setting uniform: " << name << "\n";
    glUniform1f(loc, value);
}

void Shader::setIntUniform(const char *name, int value) const
{
    int loc = glGetUniformLocation(program, name);
    if (loc == -1)
        std::cout << "Error setting uniform: " << name << "\n";
    glUniform1i(loc, value);
}

void Shader::use() const
{
    glUseProgram(program);
}
