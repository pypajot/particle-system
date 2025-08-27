#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in int ttl;

uniform mat4 camera;

void main()
{
    if (ttl = )
    gl_Position = camera * vec4(aPos, 1.0);
}