# LLVM Tier-2 JIT Engine build rule
LLVM_CONFIG ?= /opt/homebrew/opt/llvm/bin/llvm-config
CXX ?= clang++

REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
BUILD_DIR ?= $(REPO_ROOT)/build
TARGET ?= $(BUILD_DIR)/libqemu-tier2.dylib
SELFTEST ?= $(BUILD_DIR)/tier2-selftest
BENCH ?= $(BUILD_DIR)/tier2-bench

LLVM_CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags)
LLVM_LIBS := $(shell $(LLVM_CONFIG) --libs core orcjit native)
LLVM_SYSLIBS := $(shell $(LLVM_CONFIG) --system-libs)

all: $(TARGET)

CC ?= clang
HLE_SRC := $(REPO_ROOT)/linux-user/hle-thunks.c
HLE_HDR := $(REPO_ROOT)/linux-user/hle-thunks.h
HLE_OBJ := $(BUILD_DIR)/hle-thunks.o

$(HLE_OBJ): $(HLE_SRC) $(HLE_HDR)
	mkdir -p $(BUILD_DIR)
	$(CC) -std=c11 -O2 -fPIC -Ilinux-user -I$(REPO_ROOT)/linux-user -c -o $@ $(HLE_SRC)

$(TARGET): tier2-jit.cpp tier2-jit.h $(HLE_OBJ)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -shared -fPIC -Itcg/llvm -I. -Ilinux-user -I$(REPO_ROOT)/linux-user -o $@ tier2-jit.cpp $(HLE_OBJ) \
	    $(LLVM_CXXFLAGS) \
	    $(LLVM_LDFLAGS) \
	    $(LLVM_LIBS) \
	    $(LLVM_SYSLIBS)

# Offline walker self-test (no QEMU needed): compiles hand-built op
# streams and differential-checks them against C++ references.
selftest: $(SELFTEST)

$(SELFTEST): tier2-jit.cpp tier2-jit.h $(HLE_OBJ)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -DTIER2_JIT_SELFTEST_MAIN -Itcg/llvm -I. -Ilinux-user -I$(REPO_ROOT)/linux-user -o $@ tier2-jit.cpp $(HLE_OBJ) \
	    $(LLVM_CXXFLAGS) \
	    $(LLVM_LDFLAGS) \
	    $(LLVM_LIBS) \
	    $(LLVM_SYSLIBS)

# Walker benchmark: 8M-iteration loop through the op walker vs scalar C++.
bench: $(BENCH)

$(BENCH): tier2-jit.cpp tier2-jit.h $(HLE_OBJ)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -DTIER2_JIT_BENCH_MAIN -Itcg/llvm -I. -Ilinux-user -I$(REPO_ROOT)/linux-user -o $@ tier2-jit.cpp $(HLE_OBJ) \
	    $(LLVM_CXXFLAGS) \
	    $(LLVM_LDFLAGS) \
	    $(LLVM_LIBS) \
	    $(LLVM_SYSLIBS)

clean:
	rm -f $(TARGET) $(SELFTEST) $(BENCH)

.PHONY: all selftest bench clean
