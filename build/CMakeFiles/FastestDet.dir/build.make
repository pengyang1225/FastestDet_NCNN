# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build

# Include any dependencies generated for this target.
include CMakeFiles/FastestDet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FastestDet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FastestDet.dir/flags.make

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o: CMakeFiles/FastestDet.dir/flags.make
CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o: ../src/FastestDet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o -c /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/src/FastestDet.cpp

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FastestDet.dir/src/FastestDet.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/src/FastestDet.cpp > CMakeFiles/FastestDet.dir/src/FastestDet.cpp.i

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FastestDet.dir/src/FastestDet.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/src/FastestDet.cpp -o CMakeFiles/FastestDet.dir/src/FastestDet.cpp.s

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.requires:

.PHONY : CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.requires

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.provides: CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.requires
	$(MAKE) -f CMakeFiles/FastestDet.dir/build.make CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.provides.build
.PHONY : CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.provides

CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.provides.build: CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o


# Object files for target FastestDet
FastestDet_OBJECTS = \
"CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o"

# External object files for target FastestDet
FastestDet_EXTERNAL_OBJECTS =

../bin/FastestDet: CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o
../bin/FastestDet: CMakeFiles/FastestDet.dir/build.make
../bin/FastestDet: /usr/local/lib/libopencv_shape.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_stitching.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_superres.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_videostab.so.3.2.0
../bin/FastestDet: ../lib/libncnn.a
../bin/FastestDet: /usr/local/lib/libopencv_objdetect.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_calib3d.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_features2d.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_flann.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_highgui.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_ml.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_photo.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_video.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_videoio.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_imgproc.so.3.2.0
../bin/FastestDet: /usr/local/lib/libopencv_core.so.3.2.0
../bin/FastestDet: CMakeFiles/FastestDet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/FastestDet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FastestDet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FastestDet.dir/build: ../bin/FastestDet

.PHONY : CMakeFiles/FastestDet.dir/build

CMakeFiles/FastestDet.dir/requires: CMakeFiles/FastestDet.dir/src/FastestDet.cpp.o.requires

.PHONY : CMakeFiles/FastestDet.dir/requires

CMakeFiles/FastestDet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FastestDet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FastestDet.dir/clean

CMakeFiles/FastestDet.dir/depend:
	cd /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build /media/py/8ee085fc-6bf0-41db-8bee-dbf383701c5c/code/git/FastestDet_NCNN/build/CMakeFiles/FastestDet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FastestDet.dir/depend
