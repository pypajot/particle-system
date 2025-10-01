# Particle system

The goal of this project was to simulate and display a high number of particle (several millions) using OpenGL and CUDA.

## Requirements

- OpenGL 4.6

- GLFW

- GLAD, installed locally using the `make glad` command

It is compiled assuming CUDA compute capabilities 8.X. on an Ubuntu system.

### Glad link

If the link in the Makefile expired, a new one can be generated with
https://gen.glad.sh/#generator=c&api=gl%3D4.6&profile=gl%3Dcore%2Cgles1%3Dcommon