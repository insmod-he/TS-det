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

# Utility rule file for runtest.

# Include the progress variables for this target.
include src/caffe/test/CMakeFiles/runtest.dir/progress.make

src/caffe/test/CMakeFiles/runtest:
	/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/test/test.testbin --gtest_shuffle

runtest: src/caffe/test/CMakeFiles/runtest
runtest: src/caffe/test/CMakeFiles/runtest.dir/build.make
.PHONY : runtest

# Rule to build all files generated by this target.
src/caffe/test/CMakeFiles/runtest.dir/build: runtest
.PHONY : src/caffe/test/CMakeFiles/runtest.dir/build

src/caffe/test/CMakeFiles/runtest.dir/clean:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/src/caffe/test && $(CMAKE_COMMAND) -P CMakeFiles/runtest.dir/cmake_clean.cmake
.PHONY : src/caffe/test/CMakeFiles/runtest.dir/clean

src/caffe/test/CMakeFiles/runtest.dir/depend:
	cd /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/src/caffe/test /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/src/caffe/test /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/src/caffe/test/CMakeFiles/runtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/caffe/test/CMakeFiles/runtest.dir/depend

