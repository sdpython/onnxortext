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

message("-- ONNXRUNTIME_LIB_DIR=${ONNXRUNTIME_LIB_DIR}")
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
message("++ onnxortext_static")

target_include_directories(onnxortext_static PUBLIC
    ${PROJECT_SOURCE_DIR}/ccode/include
    ${PROJECT_SOURCE_DIR}/ccode/include/onnxruntime
    ${PROJECT_SOURCE_DIR}/ccode/source)

set(STATIC_COMPILE_DEFINITIONS "PYTHON_OP_SUPPORT")
target_compile_definitions(onnxortext_static PRIVATE ${STATIC_COMPILE_DEFINITIONS})
set(STATIC_LIBRARIES onnxortext_static)

###################
# unit tests static
###################

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
message(STATUS "fetch CPack")
include(CPack)

# Enable CTest
enable_testing()
message(STATUS "fetch CTest")
include(CTest)

set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/_unittests)
message(STATUS "fetch googletest")
include(googletest)
file(GLOB STATIC_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")
add_executable(onnxortext_static_test ${STATIC_TEST_SRC})
message("++ onnxortext_static_test")
target_link_libraries(onnxortext_static_test PRIVATE gtest_main ${STATIC_LIBRARIES})
target_include_directories(onnxortext_static_test PUBLIC
    ${PROJECT_SOURCE_DIR}/ccode/include
    ${PROJECT_SOURCE_DIR}/ccode/include/onnxruntime
    ${PROJECT_SOURCE_DIR}/ccode/source)

add_test(NAME onnxortext_static_test COMMAND $<TARGET_FILE:onnxortext_static_test>)

#########
# DYNAMIC
#########

file(GLOB TARGET_SRC_DLL "./ccodedll/*.*")

if (WIN32)
    list(APPEND TARGET_SRC_DLL "${PROJECT_SOURCE_DIR}/ccodedll/dll_onnxortext.def")
endif()

message("++ onnxortext")
add_library(onnxortext SHARED ${TARGET_SRC_DLL})
target_link_libraries(onnxortext PRIVATE ${STATIC_LIBRARIES})
target_include_directories(onnxortext PRIVATE
    ${PROJECT_SOURCE_DIR}/ccode/include
    ${PROJECT_SOURCE_DIR}/ccode/include/onnxruntime
    ${PROJECT_SOURCE_DIR}/ccode/source)

################
# unit tests ops
################

file(GLOB OPS_TEST_SRC "${TEST_SRC_DIR}/ops_test/*.*")
add_executable(onnxortext_ops_test ${OPS_TEST_SRC})
message("++ onnxortext_ops_test")
target_include_directories(onnxortext_ops_test PRIVATE
    ${PROJECT_SOURCE_DIR}/ccode/include
    ${PROJECT_SOURCE_DIR}/ccode/include/onnxruntime
    ${PROJECT_SOURCE_DIR}/ccode/source
    ${PROJECT_SOURCE_DIR}/ccodedll)

if(ONNXRUNTIME_LIB_DIR)
    target_link_directories(onnxortext_ops_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
    target_link_libraries(onnxortext_ops_test onnxortext onnxruntime gtest_main ${STATIC_LIBRARIES})
    if (WIN32)
      file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
      file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
      add_custom_command(
        TARGET onnxortext_ops_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:onnxortext_ops_test>)
    endif()
endif()

set(TEST_DATA_SRC ${TEST_SRC_DIR}/data)
set(TEST_DATA_DES ${onnxortext_BINARY_DIR}/data)

# Copy test data from source to destination.
add_custom_command(
    TARGET onnxortext_ops_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC} ${TEST_DATA_DES})
    add_test(NAME onnxortext_ops_test COMMAND $<TARGET_FILE:onnxortext_ops_test>)
