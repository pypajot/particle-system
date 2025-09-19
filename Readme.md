# Particle system

The goal of this project was to simulate and display a high number of particle (several millions) using OpenGL and CUDA.

## Requirements


The project uses GLAD, GLFW and OpenGL 4.6 Core.

It is compiled assuming CUDA compute capabilities 8.X. on an Ubuntu system.

### Glad link

If glad is not installed on the machine, it can be installed locally via the `make glad` command

If the link in the Makefile expired, a new one can be generated with
https://gen.glad.sh/#generator=c&api=gl%3D4.6&profile=gl%3Dcore%2Cgles1%3Dcommon