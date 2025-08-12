#pragma once

class Shader
{
    public:
        int program;

        Shader();
        Shader(const char *vertexPath, const char *fragmentPath);
        Shader(Shader &other);
        ~Shader();

        Shader operator=(Shader &other);
        
        void use();

        void setFloatUniform(const char *name, float value);
};

std::string loadShader(std::string shaderPath);
int compileShader(std::string shaderSourceString, int shaderType);