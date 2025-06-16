# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(basic_code VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(OPENAL_LIBRARY ${PROJECT_SOURCE_DIR}/../SFML/extlibs/libs-msvc/x64/openal32.lib)
# Set the build type to Release if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add source files
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/code/main.cpp)


# Add the executable
add_executable(basic_code ${SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/../SFML/include)

link_directories(${PROJECT_SOURCE_DIR}/../SFML/lib)

# Link the executable to the libraries in the lib directory
target_link_libraries(basic_code PUBLIC sfml-graphics sfml-system sfml-window)

set_target_properties(
    basic_code PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${COMMON_OUTPUT_DIR}/bin"
)


