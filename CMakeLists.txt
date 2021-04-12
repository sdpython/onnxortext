cmake_minimum_required(VERSION 3.16.0)
project(onnxortext VERSION 0.1.0 LANGUAGES C CXX)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - using RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)

########
# STATIC
########

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")

if ((NOT ENABLE_PYTHON) AND (NOT ONNXRUNTIME))
    message(FATAL_ERROR "Cannot find onnxruntime in the default library paths, please specify the ONNXRUNTIME_LIB_DIR.")
endif()

if(NOT CC_OPTIMIZE)
    message("!!!THE COMPILER OPTIMIZATION HAS BEEN DISABLED, DEBUG-ONLY!!!")
    string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

    if (NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Od")
    endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "...")
if(NOT "${CMAKE_FIND_FRAMEWORK}" STREQUAL "NEVER")
    message(FATAL_ERROR "CMAKE_FIND_FRAMEWORK is not NEVER")
endif()

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals)
include(FetchContent)

file(GLOB_RECURSE TARGET_SRC "./ccode/*.*")
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_library(onnxortext_static STATIC ${TARGET_SRC})

target_include_directories(onnxortext_static PUBLIC
    ${PROJECT_SOURCE_DIR}/ccode/include
    ${PROJECT_SOURCE_DIR}/ccode/include/onnxruntime
    ${PROJECT_SOURCE_DIR}/ccode/shared
    ${PROJECT_SOURCE_DIR}/ccode/source)

set(STATIC_COMPILE_DEFINITIONS "")

target_compile_definitions(
    onnxortext_static PRIVATE ${STATIC_COMPILE_DEFINITIONS})

set(static_libraries ${onnxortext_static})

########
# Python
########

file(GLOB SHARED_TARGET_SRC "./ccode/shared/*.cc" "./ccode/shared/*.h")
file(GLOB TARGET_SRC_PY "./ccodepy/*.*")

set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
    message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
endif()
find_package(Python3 COMPONENTS Interpreter Development)

        #if (WIN32)
        #    list(APPEND shared_TARGET_SRC 
        #        "${PROJECT_SOURCE_DIR}/onnxruntime_customops/onnxortext.def")
        # endif()

Python3_add_library(onnxortext SHARED ${TARGET_SRC_PY} ${SHARED_TARGET_SRC})
list(APPEND COMPILE_DEFINITIONS PYTHON_OP_SUPPORT)
target_compile_definitions(onnxortext PRIVATE ${COMPILE_DEFINITIONS})
target_link_libraries(onnxortext PRIVATE ${static_libraries})

# Add dependencies to pybind11, numpy, python.

message(STATUS "fetch pybind11")
include(pybind11)
set(NUMPY_NOT_FOUND false)
exec_program("${Python3_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
          "Cannot get NumPy include directory: Is NumPy installed?")
endif(${NUMPY_NOT_FOUND})

target_include_directories(onnxortext PRIVATE ${NUMPY_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS})

############
# unit tests
############

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
message(STATUS "fetch CPack")
include(CPack)

# Enable CTest
enable_testing()
message(STATUS "fetch CTest")
include(CTest)

set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test)
message(STATUS "fetch googletest")
include(googletest)
file(GLOB STATIC_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")
add_executable(onnxortext_static_test ${static_TEST_SRC})

target_link_libraries(onnxortext_static_test gtest_main ${static_libraries})
add_test(NAME onnxortext_static_test COMMAND $<TARGET_FILE:onnxortext_static_test>)

file(GLOB SHARED_TEST_SRC "${TEST_SRC_DIR}/shared_test/*.cc")
add_executable(onnxortext_test ${SHARED_TEST_SRC})

if (ONNXRUNTIME_LIB_DIR)
    target_link_directories(onnxortext_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
    target_link_libraries(onnxortext_test onnxortext onnxruntime gtest_main ${static_libraries})
    if (WIN32)
      file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
      file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
      add_custom_command(
        TARGET onnxortext_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:onnxortext_test>)
    endif()
endif()

set(TEST_DATA_SRC ${TEST_SRC_DIR}/data)
set(TEST_DATA_DES ${onnxortext_BINARY_DIR}/data)

# Copy test data from source to destination.
add_custom_command(
    TARGET onnxortext_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC} ${TEST_DATA_DES})
    add_test(NAME onnxortext_test COMMAND $<TARGET_FILE:onnxortext_test>)
