# Install script for directory: /data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/draw_net.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/detect.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/classify.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/requirements.txt"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/io.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/net_spec.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/detector.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/__init__.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/classifier.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/draw.py"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/pycaffe.py"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         RPATH "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/install/lib:/usr/local/lib:/usr/local/cuda-7.0/lib64")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/lib/_caffe.so")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    FILE(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so"
         OLD_RPATH "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/lib:/usr/local/lib:/usr/local/cuda-7.0/lib64::::::::"
         NEW_RPATH "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/install/lib:/usr/local/lib:/usr/local/cuda-7.0/lib64")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe.so")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/imagenet"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/proto"
    "/data2/HongliangHe/Tsinghua_Tencent_100K/HardMining_caffe/caffe/python/caffe/test"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

