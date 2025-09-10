#version 460 core
out vec4 FragColor;

uniform float cursorX;
uniform float cursorY;
uniform float height;
uniform float near;
uniform float far;
uniform float mouseDepth;
uniform float frameTimeX;
uniform float frameTimeY;
uniform float frameTimeZ;

void main()
{
    float distX = (gl_FragCoord.x - cursorX) / height;
    float distY = (height - gl_FragCoord.y - cursorY) / height;
    float distZ = near * far / (gl_FragCoord.z * (near - far) + far) - mouseDepth;	

    float dist = sqrt(distX * distX + distY * distY + distZ * distZ);
    FragColor = vec4(frameTimeX / dist, frameTimeY / dist, frameTimeZ / dist, 1.0f);
} 