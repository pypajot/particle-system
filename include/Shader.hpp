#pragma once

class Shader
{
    public:
        int program;

        Shader();
        Shader(const char *vertexPath, const char *fragmentPath);
        Shader(const Shader &other);
        ~Shader();

        Shader operator=(const Shader &other);
        
        void use() const;

        void setFloatUniform(const char *name, float value) const;
        void setIntUniform(const char *name, int value) const;
};