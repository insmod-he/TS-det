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
include tools/CMakeFiles/convert_annotation_data.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/convert_annotation_data.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/convert_annotation_data.dir/flags.make

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o: tools/CMakeFiles/convert_annotation_data.dir/flags.make
tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o: tools/convert_annotation_data.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o -c /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/convert_annotation_data.cpp

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.i"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/convert_annotation_data.cpp > CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.i

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.s"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/convert_annotation_data.cpp -o CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.s

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.requires:
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.requires

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.provides: tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/convert_annotation_data.dir/build.make tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.provides.build
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.provides

tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.provides.build: tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o

# Object files for target convert_annotation_data
convert_annotation_data_OBJECTS = \
"CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o"

# External object files for target convert_annotation_data
convert_annotation_data_EXTERNAL_OBJECTS =

tools/convert_annotation_data: tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o
tools/convert_annotation_data: tools/CMakeFiles/convert_annotation_data.dir/build.make
tools/convert_annotation_data: lib/libcaffe.so
tools/convert_annotation_data: lib/libproto.a
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libglog.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/convert_annotation_data: /usr/local/lib/libprotobuf.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libglog.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/convert_annotation_data: /usr/local/lib/libprotobuf.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libhdf5.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/convert_annotation_data: /usr/lib/libsnappy.so
tools/convert_annotation_data: /usr/local/cuda-7.0/lib64/libcudart.so
tools/convert_annotation_data: /usr/local/cuda-7.0/lib64/libcurand.so
tools/convert_annotation_data: /usr/local/cuda-7.0/lib64/libcublas.so
tools/convert_annotation_data: /usr/local/cuda-7.0/lib64/libcudnn.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
tools/convert_annotation_data: /usr/lib/liblapack_atlas.so
tools/convert_annotation_data: /usr/lib/libcblas.so
tools/convert_annotation_data: /usr/lib/libatlas.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/convert_annotation_data: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/convert_annotation_data: tools/CMakeFiles/convert_annotation_data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable convert_annotation_data"
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert_annotation_data.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/convert_annotation_data.dir/build: tools/convert_annotation_data
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/build

tools/CMakeFiles/convert_annotation_data.dir/requires: tools/CMakeFiles/convert_annotation_data.dir/convert_annotation_data.cpp.o.requires
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/requires

tools/CMakeFiles/convert_annotation_data.dir/clean:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools && $(CMAKE_COMMAND) -P CMakeFiles/convert_annotation_data.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/clean

tools/CMakeFiles/convert_annotation_data.dir/depend:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/tools/CMakeFiles/convert_annotation_data.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/convert_annotation_data.dir/depend

