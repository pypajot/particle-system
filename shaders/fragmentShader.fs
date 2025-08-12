#version 450 core
out vec4 FragColor;

// uniform float cursorX;
// uniform float cursorY;
// uniform float height;
// uniform float width;
// uniform float frameTimeX;
// uniform float frameTimeY;
// uniform float frameTimeZ;

void main()
{
    // float distX = (gl_FragCoord.x - cursorX) / width;
    // float distY = (-gl_FragCoord.y - cursorY) / height;
    // float distZ = gl_FragCoord.z;
    // float dist = 1.0f;
    // FragColor = vec4(frameTimeX / dist, frameTimeY / dist, frameTimeZ / dist, 1.0f);
    FragColor = vec4(0.0f, 1.0f, 1.0f, 1.0f);
} 