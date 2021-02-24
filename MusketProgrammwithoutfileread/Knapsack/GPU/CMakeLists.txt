	cmake_minimum_required(VERSION 3.10)
	project(Knapsack VERSION 1.0.0 LANGUAGES CXX)
	
	# required macros
	SET( CMAKE_CXX_FLAGS_DEV "-g -O0 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during DEV builds." FORCE )
	SET( CMAKE_CXX_FLAGS_TEST "-gopt -fast pinned -O4 -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during TEST builds." FORCE )
	SET( CMAKE_CXX_FLAGS_VTUNE "-gopt -fast -O4 -w" CACHE STRING "Flags used by the C++ compiler during VTUNE builds." FORCE )
	SET( CMAKE_CXX_FLAGS_NVPROF "-gopt -fast -O4 -w -Minfo=accel" CACHE STRING "Flags used by the C++ compiler during NVPROF builds." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARK "-fast -O4 -ta:tesla:cc60,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARKPALMA "-fast -O4 -tp -ta:tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Palma." FORCE )
	SET( CMAKE_CXX_FLAGS_BENCHMARKTAURUS "-fast -O4 -tp=haswell -ta=tesla:cc35,pinned -w" CACHE STRING "Flags used by the C++ compiler during Benchmark builds for Taurus." FORCE )
		
	# output path for binaries and libraries
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
	
	# packages
	
	find_package(OpenMP REQUIRED)
	
	find_package(OpenACC REQUIRED)

	
	add_executable(Knapsack_0 ${PROJECT_SOURCE_DIR}/src/Knapsack_0.cpp)
	target_compile_features(Knapsack_0 PRIVATE cxx_std_14)
	target_include_directories(Knapsack_0 PRIVATE ${PROJECT_SOURCE_DIR}/include)
	target_compile_definitions(Knapsack_0 PRIVATE)
	target_compile_options(Knapsack_0 PRIVATE ${OpenMP_CXX_FLAGS} ${OpenACC_CXX_FLAGS})
	target_link_libraries(Knapsack_0 PRIVATE ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_LIBRARIES} ${OpenACC_CXX_FLAGS})
