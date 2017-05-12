# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe

# Include any dependencies generated for this target.
include tools/CMakeFiles/finetune_net.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/finetune_net.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/finetune_net.dir/flags.make

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/CMakeFiles/finetune_net.dir/flags.make
tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/finetune_net.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/finetune_net.dir/finetune_net.cpp.o -c /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/finetune_net.cpp

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finetune_net.dir/finetune_net.cpp.i"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/finetune_net.cpp > CMakeFiles/finetune_net.dir/finetune_net.cpp.i

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finetune_net.dir/finetune_net.cpp.s"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/finetune_net.cpp -o CMakeFiles/finetune_net.dir/finetune_net.cpp.s

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires:
.PHONY : tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/finetune_net.dir/build.make tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides.build
.PHONY : tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides.build: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o

# Object files for target finetune_net
finetune_net_OBJECTS = \
"CMakeFiles/finetune_net.dir/finetune_net.cpp.o"

# External object files for target finetune_net
finetune_net_EXTERNAL_OBJECTS =

tools/finetune_net: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/build.make
tools/finetune_net: lib/libcaffe.so
tools/finetune_net: lib/libproto.a
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libglog.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/finetune_net: /usr/local/lib/libprotobuf.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libglog.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/finetune_net: /usr/local/lib/libprotobuf.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/finetune_net: /usr/lib/libsnappy.so
tools/finetune_net: /usr/local/cuda-7.0/lib64/libcudart.so
tools/finetune_net: /usr/local/cuda-7.0/lib64/libcurand.so
tools/finetune_net: /usr/local/cuda-7.0/lib64/libcublas.so
tools/finetune_net: /usr/local/cuda-7.0/lib64/libcudnn.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
tools/finetune_net: /usr/lib/liblapack_atlas.so
tools/finetune_net: /usr/lib/libcblas.so
tools/finetune_net: /usr/lib/libatlas.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable finetune_net"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finetune_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/finetune_net.dir/build: tools/finetune_net
.PHONY : tools/CMakeFiles/finetune_net.dir/build

tools/CMakeFiles/finetune_net.dir/requires: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires
.PHONY : tools/CMakeFiles/finetune_net.dir/requires

tools/CMakeFiles/finetune_net.dir/clean:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && $(CMAKE_COMMAND) -P CMakeFiles/finetune_net.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/finetune_net.dir/clean

tools/CMakeFiles/finetune_net.dir/depend:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/CMakeFiles/finetune_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/finetune_net.dir/depend

