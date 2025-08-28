NAME := particle
CC := c++
GLFLAGS := -lGL -lGLU -lglfw -lX11 -lXxf86vm -lXrandr -lpthread -lXi
CPPFLAGS := -Wall -Wextra -Werror -g -MMD --std=c++20

OBJDIR := objs
SRCDIR := srcs

SRCS := main.cpp Window.cpp AEngine.cpp EngineStatic.cpp EngineGen.cpp Shader.cpp Camera.cpp

OBJS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))
DEPS := $(patsubst %.cpp,$(OBJDIR)/%.d,$(SRCS))

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


all : $(NAME)

$(NAME): $(OBJS) Makefile
	$(CC) $(CPPFLAGS) -o $@ $(OBJS) $(GLFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CC) $(CPPFLAGS) -o $@ -c $< -I$(INCS)

clean:
	rm -rfd $(OBJDIR)

fclean: clean
	rm -f $(NAME)

re: fclean all
