cmake_minimum_required(VERSION 3.20)

project(cs336_tokenizer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "pybind11 include dir: ${PYBIND11_INCLUDE_DIR}")

add_library(cpp_tokenizer MODULE
        cs336_basics/cpp_tokenizer.cpp
)

target_include_directories(cpp_tokenizer PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR}
)

target_link_libraries(cpp_tokenizer PRIVATE
        Python3::Python
)

set_target_properties(cpp_tokenizer PROPERTIES
        PREFIX ""
        SUFFIX ".so"
)