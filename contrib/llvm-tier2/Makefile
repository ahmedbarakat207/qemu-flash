# Tier-2 prototype harness: builds tier2-demo against the system LLVM
# (Homebrew LLVM on macOS, or distro LLVM on Linux).
LLVM_CONFIG ?= /opt/homebrew/opt/llvm/bin/llvm-config
CXX ?= c++

tier2-demo: tier2-demo.cpp
	$(CXX) -std=c++17 -O2 -o $@ $< \
	    $(shell $(LLVM_CONFIG) --cxxflags) \
	    $(shell $(LLVM_CONFIG) --ldflags) \
	    $(shell $(LLVM_CONFIG) --libs core orcjit native) \
	    $(shell $(LLVM_CONFIG) --system-libs)

op-run: op-run.cpp
	$(CXX) -std=c++17 -O2 -o $@ $< \
	    $(shell $(LLVM_CONFIG) --cxxflags) \
	    $(shell $(LLVM_CONFIG) --ldflags) \
	    $(shell $(LLVM_CONFIG) --libs core orcjit native) \
	    $(shell $(LLVM_CONFIG) --system-libs)

.PHONY: run clean
run: tier2-demo
	./tier2-demo ssa
	./tier2-demo env

clean:
	rm -f tier2-demo op-run
