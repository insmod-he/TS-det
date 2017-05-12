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
include examples/CMakeFiles/classification.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/classification.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/classification.dir/flags.make

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o: examples/CMakeFiles/classification.dir/flags.make
examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o: examples/cpp_classification/classification.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/classification.dir/cpp_classification/classification.cpp.o -c /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/cpp_classification/classification.cpp

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classification.dir/cpp_classification/classification.cpp.i"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/cpp_classification/classification.cpp > CMakeFiles/classification.dir/cpp_classification/classification.cpp.i

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classification.dir/cpp_classification/classification.cpp.s"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/cpp_classification/classification.cpp -o CMakeFiles/classification.dir/cpp_classification/classification.cpp.s

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.requires:
.PHONY : examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.requires

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.provides: examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/classification.dir/build.make examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.provides.build
.PHONY : examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.provides

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.provides.build: examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o

# Object files for target classification
classification_OBJECTS = \
"CMakeFiles/classification.dir/cpp_classification/classification.cpp.o"

# External object files for target classification
classification_EXTERNAL_OBJECTS =

examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o
examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/build.make
examples/cpp_classification/classification: lib/libcaffe.so
examples/cpp_classification/classification: lib/libproto.a
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libglog.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/cpp_classification/classification: /usr/local/lib/libprotobuf.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libglog.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/cpp_classification/classification: /usr/local/lib/libprotobuf.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libhdf5.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/liblmdb.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libleveldb.so
examples/cpp_classification/classification: /usr/lib/libsnappy.so
examples/cpp_classification/classification: /usr/local/cuda-7.0/lib64/libcudart.so
examples/cpp_classification/classification: /usr/local/cuda-7.0/lib64/libcurand.so
examples/cpp_classification/classification: /usr/local/cuda-7.0/lib64/libcublas.so
examples/cpp_classification/classification: /usr/local/cuda-7.0/lib64/libcudnn.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
examples/cpp_classification/classification: /usr/lib/liblapack_atlas.so
examples/cpp_classification/classification: /usr/lib/libcblas.so
examples/cpp_classification/classification: /usr/lib/libatlas.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libpython2.7.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_python.so
examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cpp_classification/classification"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/classification.dir/link.txt --verbose=$(VERBOSE)
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && ln -sf /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/cpp_classification/classification /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/cpp_classification/classification.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/classification.dir/build: examples/cpp_classification/classification
.PHONY : examples/CMakeFiles/classification.dir/build

examples/CMakeFiles/classification.dir/requires: examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o.requires
.PHONY : examples/CMakeFiles/classification.dir/requires

examples/CMakeFiles/classification.dir/clean:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples && $(CMAKE_COMMAND) -P CMakeFiles/classification.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/classification.dir/clean

examples/CMakeFiles/classification.dir/depend:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/examples/CMakeFiles/classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/classification.dir/depend

