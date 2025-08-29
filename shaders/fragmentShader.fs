#version 460 core
out vec4 FragColor;

uniform float cursorX;
uniform float cursorY;
uniform float height;
uniform float near;
uniform float far;
uniform float frameTimeX;
uniform float frameTimeY;
uniform float frameTimeZ;

void main()
{
    float distX = (gl_FragCoord.x - cursorX) / height;
    float distY = (height - gl_FragCoord.y - cursorY) / height;
    float z = gl_FragCoord.z * 2.0 - 1.0; // back to NDC 
    float distZ = ((2.0 * near * far) / (far + near - z * (far - near)) - 2);	
    float dist = distX * distX + distY * distY + distZ * distZ;
    dist = sqrt(dist) * 2;
    FragColor = vec4(frameTimeX / dist, frameTimeY / dist, frameTimeZ / dist, 1.0f);
} 