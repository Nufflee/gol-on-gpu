CXX=cl
NVCC=nvcc

BUILD_DIR=build

# C++ compiler
CXX_INCLUDES=/I %%VCPKG_ROOT%%\installed\x64-windows\include /I "%%CUDA_PATH%%\include"
CXX_LIB_PATHS=/LIBPATH:%%VCPKG_ROOT%%\installed\x64-windows\lib /LIBPATH:%%VCPKG_ROOT%%\installed\x64-windows\lib\manual-link /LIBPATH:"%%CUDA_PATH%%\lib\x64"
CXX_LIBS=SDL2.lib SDL2main.lib shell32.lib cudart.lib

CXX_FLAGS=/EHsc /Fe: gol.exe /Fo: $(BUILD_DIR)/ /Fd: $(BUILD_DIR)/ /DEBUG /Ox /Z7 $(CXX_INCLUDES)
LD_FLAGS=/SUBSYSTEM:CONSOLE /NODEFAULTLIB:msvcrt.lib $(CXX_LIB_PATHS) $(CXX_LIBS)

CXX_SOURCES=src/*.cpp

# NVCC compiler
NVCC_FLAGS=-Xcompiler "/EHsc" -c -allow-unsupported-compiler -O3 -g

NVCC_SOURCES=src/*.cu

all: $(CXX_SOURCES) gpu_game_of_life.obj
	$(CXX) $(CXX_SOURCES) ./$(BUILD_DIR)/gpu_game_of_life.obj $(CXX_FLAGS) /link $(LD_FLAGS)

gpu_game_of_life.obj: $(NVCC_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/$@ $**

clean:
	rm build/*