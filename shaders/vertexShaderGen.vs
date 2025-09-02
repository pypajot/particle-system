#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in float ttl;

uniform float maxTtl;
uniform mat4 camera;

void main()
{
    gl_Position = camera * vec4(aPos, 1.0);
    if (ttl >= maxTtl)
        gl_Position = vec4(1.0f, 1.0f, 1.0f, 0.0f);

}