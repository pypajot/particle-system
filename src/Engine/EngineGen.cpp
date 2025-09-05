#include <random>
#include <cmath>

#include "Engine/EngineGen.hpp"


EngineGen::EngineGen() : AEngine()
{
}

EngineGen::EngineGen(int particleQuantity) : AEngine(particleQuantity), worker(VBO, particleQty)
{
    initType = ENGINE_INIT_GEN;    
    vertexPath = "shaders/vertexShaderGen.vs";
    shader = Shader(vertexPath.c_str(), fragmentPath.c_str());
    particlePerFrame = BASE_PPF;

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBufferData(GL_ARRAY_BUFFER, particleQty * 7 * sizeof(float), 0, GL_STREAM_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    reset();
}

EngineGen::EngineGen(const EngineGen &other) : AEngine(other)
{
    particlePerFrame = other.particlePerFrame;
    generatorOn = other.generatorOn;
    worker = other.worker;
}

EngineGen::~EngineGen()
{
}

EngineGen &EngineGen::operator=(const EngineGen &other)
{
    if (this == &other)
        return *this;
        
    AEngine::operator=(other);
    particlePerFrame = other.particlePerFrame;
    generatorOn = other.generatorOn;
    worker = other.worker;
    return *this;
}

void EngineGen::reset()
{
    camera.resetPosition();
    generatorOn = true;
    worker->initGen();
    simulationOn = false;
}

void EngineGen::useShader(float frameTime, float cursorX, float cursorY, float height)
{
    mat4 toScreen = camera.coordToScreenMatrix();
    int camLoc = glGetUniformLocation(shader.program, "camera");
    
    glUniformMatrix4fv(camLoc, 1, GL_FALSE, &toScreen.value[0][0]);
    shader.setFloatUniform("maxTtl", timeToLive);
    shader.setFloatUniform("frameTimeX", (1 + sin(frameTime)) / 2);
    shader.setFloatUniform("frameTimeY", (1 + sin(frameTime + 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("frameTimeZ", (1 + sin(frameTime - 2 * M_PI / 3)) / 2);
    shader.setFloatUniform("cursorX", cursorX);
    shader.setFloatUniform("cursorY", cursorY);
    shader.setFloatUniform("height", height);
    shader.setFloatUniform("near", camera.near);
    shader.setFloatUniform("far", camera.far);
    shader.setFloatUniform("mouseDepth", mouseDepth);
    shader.use();
}

void EngineGen::run()
{
    engine->camera.move();
    
    if (!simulationOn)
        return;

    worker.Map();
    if (generatorOn)
        worker->generate(particlePerFrame);
    worker->call(gravityPos, gravityOn, gravityStrength);
    worker.Unmap();
}

void EngineGen::ppfUp()
{
    if (particlePerFrame >= MAX_PPF)
    {
        std::cout << "Particle per frame at max value : " << particlePerFrame << "\n";
        return;
    }
    particlePerFrame += 500;
    std::cout << "Particle per frame increased, new value : " << particlePerFrame << "\n";
}

void EngineGen::ppfDown()
{
    if (particlePerFrame <= MIN_PPF)
    {
        std::cout << "Particle per frame at min value : " << particlePerFrame << "\n";
        return;
    }   
    particlePerFrame -= 500;
    std::cout << "Particle per frame decreased, new value : " << particlePerFrame << "\n";
}
