cmake_minimum_required(VERSION 3.18)
project(operators C CXX)

option(ENABLE_CPU "Enable or disable cpu kernel" ON)
option(ENABLE_NV_GPU "Enable or disable Nvidia GPU kernel" ON)
option(DEBUG "Enable or disable debug" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
add_definitions(-DDEBUG_MODE)

file(GLOB SRC_FILES
    "src/main.c"
    "src/ops/*/operator.cc"
)
add_library(operators SHARED ${SRC_FILES})
if (ENABLE_CPU)
    add_definitions(-DENABLE_CPU)
    file(GLOB SRC_FILES_CPU
        "src/devices/cpu/*.cc"
        "src/ops/*/cpu/*.cc"
    )
    target_link_libraries(operators PRIVATE cpu)
    add_library(cpu STATIC ${SRC_FILES_CPU})
    set_target_properties(cpu PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    if (NOT WIN32)
        set_target_properties(cpu PROPERTIES
            COMPILE_FLAGS "-fPIC"
        )
    endif()
endif()

if (ENABLE_NV_GPU)
    add_definitions(-DENABLE_NV_GPU)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    enable_language(CUDA)

    file(GLOB SRC_FILES_NV_GPU
        "src/ops/*/cuda/*.cu"
    )
    target_link_libraries(operators PRIVATE nv_gpu CUDA::cublas)
    set_target_properties(operators PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;80"
    )
    add_library(nv_gpu STATIC ${SRC_FILES_NV_GPU})
    set_target_properties(nv_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;80"
    )
    if (WIN32)
        target_compile_options(nv_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
        )
    else()
        target_compile_options(nv_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
        )
    endif()
endif()

add_executable(main ${SRC_FILES})
target_link_libraries(main PRIVATE operators)
