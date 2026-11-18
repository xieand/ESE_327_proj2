# Four classifiers, each built three ways from one algorithm description:
#
#   make cpu   -> bin/<name>_cpu       sequential C++ (any C++17 compiler)
#   make gpu   -> bin/<name>_gpu       CUDA (needs nvcc)
#   make emu   -> bin/<name>_gpu_emu   the CUDA source compiled as plain C++
#                                      (serial kernel emulation, for testing
#                                      on machines without an NVIDIA GPU)
#   make tools -> bin/dataset_info     schema / class / split report for a table
#   make       -> cpu + emu + tools, plus gpu when nvcc is on the PATH
#
# <name> is dtree (decision tree induction: gain, gain ratio, Gini), nbayes
# (naive Bayes), knn (k-nearest neighbors) or logreg (multinomial logistic
# regression).
#
# Override e.g. CUDA_ARCH=-arch=sm_86 for a specific GPU (default: native).
# On Windows nvcc needs the MSVC environment; use scripts/build_gpu.bat there.

CXX       ?= g++
NVCC      ?= nvcc
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CUDA_ARCH ?= -arch=native
# --fmad=false: no fused multiply-add, so that the kernels round exactly like
# the CPU code and both produce the same distances and scores.
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_ARCH) --fmad=false

INC        := -Iinclude
BIN        := bin
COMMON_SRC := src/dataset.cpp src/eval.cpp src/runner.cpp src/util.cpp
COMMON     := $(COMMON_SRC) include/dataset.h include/eval.h include/runner.h include/util.h
CUDA_H     := include/cuda_compat.h

# Sources and headers of every program besides its own main file and COMMON.
dtree_SRC  := src/dtree.cpp
dtree_DEP  := include/dtree.h include/dtree_app.h
nbayes_SRC :=
nbayes_DEP := include/nbayes.h
knn_SRC    := src/encode.cpp
knn_DEP    := include/encode.h include/knn_app.h
logreg_SRC := src/encode.cpp
logreg_DEP := include/encode.h include/logreg.h include/lbfgs.h

NAMES := dtree nbayes knn logreg

ifeq ($(OS),Windows_NT)
  EXE := .exe
  # Another MinGW runtime earlier on PATH (e.g. Git for Windows' mingw64/bin)
  # makes ld and the produced binaries load mismatched DLLs. Put the
  # compiler's own directory first and link the C++ runtime statically.
  CXX_DIR := $(dir $(shell command -v $(CXX) 2>/dev/null))
  ifneq ($(CXX_DIR),)
    export PATH := $(CXX_DIR):$(PATH)
  endif
  LDFLAGS ?= -static
endif

HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(OS),Windows_NT)
  # nvcc drives cl.exe on Windows; without the MSVC environment leave the
  # CUDA build to scripts/build_gpu.bat.
  ifeq ($(shell command -v cl 2>/dev/null),)
    HAVE_NVCC :=
  endif
endif

.PHONY: all cpu gpu emu tools clean
all: cpu emu tools $(if $(HAVE_NVCC),gpu)

cpu: $(foreach n,$(NAMES),$(BIN)/$(n)_cpu$(EXE))
emu: $(foreach n,$(NAMES),$(BIN)/$(n)_gpu_emu$(EXE))
gpu: $(foreach n,$(NAMES),$(BIN)/$(n)_gpu$(EXE))
tools: $(BIN)/dataset_info$(EXE)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/dataset_info$(EXE): src/dataset_info.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/dataset_info.cpp $(COMMON_SRC) $(LDFLAGS) -o $@

define PROGRAM
$(BIN)/$(1)_cpu$(EXE): src/$(1)_cpu.cpp $$($(1)_SRC) $$($(1)_DEP) $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/$(1)_cpu.cpp $$($(1)_SRC) $(COMMON_SRC) $(LDFLAGS) -o $$@

$(BIN)/$(1)_gpu_emu$(EXE): src/$(1)_gpu_emu.cpp src/$(1)_gpu.cu $$($(1)_SRC) $$($(1)_DEP) $(CUDA_H) $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/$(1)_gpu_emu.cpp $$($(1)_SRC) $(COMMON_SRC) $(LDFLAGS) -o $$@

$(BIN)/$(1)_gpu$(EXE): src/$(1)_gpu.cu $$($(1)_SRC) $$($(1)_DEP) $(CUDA_H) $(COMMON) | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INC) src/$(1)_gpu.cu $$($(1)_SRC) $(COMMON_SRC) -o $$@
endef
$(foreach n,$(NAMES),$(eval $(call PROGRAM,$(n))))

clean:
	rm -rf $(BIN)
