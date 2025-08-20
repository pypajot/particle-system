#version 450 core
out vec4 FragColor;

uniform float cursorX;
uniform float cursorY;
uniform float height;
uniform float frameTimeX;
uniform float frameTimeY;
uniform float frameTimeZ;

void main()
{
    float distX = (gl_FragCoord.x - cursorX) / height;
    float distY = (height - gl_FragCoord.y - cursorY) / height;

    float dist = distX * distX + distY * distY;
    dist = sqrt(dist) * 2;
    FragColor = vec4(frameTimeX / dist, frameTimeY / dist, frameTimeZ / dist, 1.0f);
} 