NAME := particle
CC := c++
GLFLAGS := -lGL -lGLU -lglfw -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lcudart
CPPFLAGS := -Wall -Wextra -Werror -g -MMD --std c++11

CUDACC := nvcc
CUDAFLAGS := -Werror all-warnings -g -MMD --std c++11 -gencode arch=compute_80,code=sm_80

OBJDIR := obj
SRCDIR := src

SRCS := main.cpp \
		Window.cpp \
		Shader.cpp \
		Camera.cpp \
		Gravity.cpp \
		FPSCounter.cpp \
		Engine/AEngine.cpp \
		Engine/EngineStatic.cpp \
		Engine/EngineGen.cpp \
		math/transform.cpp \
		math/vec3.cpp \
		math/vec4.cpp \
		math/mat4.cpp \
		glad/gl.cpp

CUDASRCS:=  Worker/AWorker.cu \
			Worker/WorkerStatic.cu \
			Worker/WorkerGen.cu \

SRCS_NODIR := $(strip $(SRCS))
OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o, $(strip $(SRCS)))
DEPS := $(patsubst %.cpp,$(OBJDIR)/%.d, $(strip $(SRCS)))

CUDAOBJS := $(patsubst %.cu,$(OBJDIR)/%.o, $(strip $(CUDASRCS)))
CUDADEPS := $(patsubst %.cu,$(OBJDIR)/%.d, $(strip $(CUDASRCS)))

INCS := ./include/

_GREY		= \033[30m
_RED		= \033[31m
_GREEN		= \033[32m
_YELLOW		= \033[33m
_BLUE		= \033[34m
_PURPLE		= \033[35m
_CYAN		= \033[36m
_WHITE		= \033[37m
_NO_COLOR	= \033[0m


all : glad $(NAME)

$(NAME): $(OBJS) $(CUDAOBJS) Makefile
	$(CC) $(CPPFLAGS) -o $(NAME) $(OBJS) $(CUDAOBJS) $(GLFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@if [ ! -d $(dir $@) ]; then \
		mkdir -p $(dir $@); \
	fi
	$(CC) $(CPPFLAGS) -o $@ -c $< -I$(INCS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@if [ ! -d $(dir $@) ]; then \
		mkdir -p $(dir $@); \
	fi
	$(CUDACC) $(CUDAFLAGS) -o $@ -c $< -I$(INCS)

-include $(DEPS)

clean:
	rm -rfd $(OBJDIR)

fclean: clean
	rm -f $(NAME)

re: fclean all

glad:
	wget "https://gen.glad.sh/generated/tmp0rc01w22glad/glad.zip" -O glad.zip
	unzip glad.zip -d glad
	mkdir -p include/glad/
	mkdir -p src/glad/
	mv glad/include/glad/gl.h include/glad/gl.h
	mv glad/src/gl.c src/glad/gl.cpp

cleanglad:
	rm -f glad.zip
	rm -rfd glad/
	rm -rfd include/glad
	rm -rfd src/glad


.PHONY: re fclean clean all cleanglad
