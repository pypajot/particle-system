NAME := particle
CC := c++
GLFLAGS := -lGL -lGLU -lglfw -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lcudart
CPPFLAGS := -Wall -Wextra -Werror -g -MMD --std c++17

CUDACC := nvcc
CUDAFLAGS := -Werror all-warnings -g -MMD --std c++17 -gencode arch=compute_80,code=sm_80

OBJDIR := objs
SRCDIR := srcs

SRCS := main.cpp \
		Window.cpp \
		AEngine.cpp \
		EngineStatic.cpp \
		EngineGen.cpp \
		Shader.cpp \
		Camera.cpp \
		math.cpp \
		vec3.cpp \
		vec4.cpp \
		mat4.cpp \
		gl.cpp

CUDASRCS:= CudaWorker.cu


OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
DEPS := $(patsubst %.cpp,$(OBJDIR)/%.d,$(SRCS))

CUDAOBJS := $(patsubst %.cu,$(OBJDIR)/%.o,$(CUDASRCS))
CUDADEPS := $(patsubst %.cu,$(OBJDIR)/%.d,$(CUDASRCS))

INCS := ./incs/

_GREY		= \033[30m
_RED		= \033[31m
_GREEN		= \033[32m
_YELLOW		= \033[33m
_BLUE		= \033[34m
_PURPLE		= \033[35m
_CYAN		= \033[36m
_WHITE		= \033[37m
_NO_COLOR	= \033[0m


all : cuda

cuda: $(OBJS) $(CUDAOBJS) Makefile
	$(CC) $(CPPFLAGS) -o $(NAME) $(OBJS) $(CUDAOBJS) $(GLFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CC) $(CPPFLAGS) -o $@ -c $< -I$(INCS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(CUDACC) $(CUDAFLAGS) -o $@ -c $< -I$(INCS)

-include $(DEPS)

clean:
	rm -rfd $(OBJDIR)

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: re fclean clean all cuda gl
