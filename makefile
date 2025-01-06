DEBUG ?= 1

SRC_DIR := .

# GPU
GPU_CARD := -arch=sm_89 # specify the proper device compute capability here

# =========== CUDA part ===========
NVCC := /usr/bin/nvcc
NVCC_FLAGS := -std=c++20 -dc -Xcompiler
ifeq ($(DEBUG),1)
  NVCC_FLAGS += -g3 -O0 -G
else
  NVCC_FLAGS += -O3
endif
CUDA_LINK_FLAGS := -dlink

CUDA_INCL := -I/usr/include
CUDA_LIBS := -L/usr/lib/x86_64-linux-gnu -lcudart 

CUDA_SRCS := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS := $(patsubst %.cu, %.o, $(notdir $(CUDA_SRCS)))

# =========== C++ part ===========
CC := g++
CFLAGS := -std=c++20 -Wall
ifeq ($(DEBUG),1)
  CFLAGS += -g3 -O0
else
  CFLAGS += -O3
endif

INCL := -I$(SRC_DIR)
LIBS :=# ARE THERE LICENSE ISSUES OF USING THESE LIBRARIES?

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst %.cpp, %.o, $(notdir $(SRCS)))

# =========== build  ===========
EXEC := test
CUDA_LINK_OBJ := cuLink.o

all: $(EXEC)
$(EXEC): $(CUDA_OBJS) $(OBJS)
ifeq ($(strip $(CUDA_OBJS)), )
	$(CC) -o $@ $(OBJS) $(LIBS)
else
	$(NVCC) $(GPU_CARD) $(CUDA_LINK_FLAGS) -o $(CUDA_LINK_OBJ) $(CUDA_OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS) $(CUDA_OBJS) $(CUDA_LINK_OBJ) $(CUDA_LIBS)
endif

%.o: %.cpp
	$(CC) $(INCL) $(CUDA_INCL) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(GPU_CARD) $(NVCC_FLAGS) $(INCL) $(CUDA_INCL) -c $< -o $@

clean:	
	rm -f $(OBJS) $(EXEC) *.o
	
.PHONY : all clean
