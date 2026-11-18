# Targets
#   make cpu   -> bin/dtree_cpu      decision tree induction (gain, gain ratio, Gini)
#   make tools -> bin/dataset_info   schema / class / split report for a table
#   make       -> everything that can be built on this machine
#
# The other classifiers and the CUDA / CUDA-emulation builds are added to
# this file as they are implemented.

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

INC        := -Iinclude
BIN        := bin
COMMON_SRC := src/dataset.cpp src/eval.cpp src/runner.cpp src/util.cpp
COMMON     := $(COMMON_SRC) include/dataset.h include/eval.h include/runner.h include/util.h
DTREE_SRC  := src/dtree.cpp
DTREE      := $(DTREE_SRC) include/dtree.h

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

.PHONY: all cpu tools clean
all: cpu tools

cpu: $(BIN)/dtree_cpu$(EXE)
tools: $(BIN)/dataset_info$(EXE)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/dataset_info$(EXE): src/dataset_info.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/dataset_info.cpp $(COMMON_SRC) $(LDFLAGS) -o $@

$(BIN)/dtree_cpu$(EXE): src/dtree_cpu.cpp $(DTREE) $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/dtree_cpu.cpp $(DTREE_SRC) $(COMMON_SRC) $(LDFLAGS) -o $@

clean:
	rm -rf $(BIN)
