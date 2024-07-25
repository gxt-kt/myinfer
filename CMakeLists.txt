cmake_minimum_required(VERSION 3.00)
project(myinfer)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

set(link_lib glog gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

aux_source_directory(./test/arma DIR_TEST_ARMA)
aux_source_directory(./test/tensor DIR_TEST_ARMA)
aux_source_directory(./test/graph DIR_TEST_ARMA)
# aux_source_directory(./test/pnnx_ir DIR_TEST_ARMA)
# aux_source_directory(./test/layer DIR_TEST_ARMA)
# aux_source_directory(./test/expression DIR_TEST_ARMA)
# aux_source_directory(./test/resnet DIR_TEST_ARMA)
# aux_source_directory(./test/yolov5 DIR_TEST_ARMA)
message(STATUS test_files:${DIR_TEST_ARMA})

aux_source_directory(./source DIR_SOURCE_ARMA)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_DETAIL_LAYER)
aux_source_directory(./source/parser DIR_PARSER)


set(test_bin_name test_${PROJECT_NAME})

add_executable(${test_bin_name} test/main.cpp ${DIR_TEST_ARMA} ${DIR_PARSER} ${DIR_SOURCE_ARMA} ${DIR_DETAIL_LAYER} ${DIR_ABSTRACT_LAYER})
target_link_libraries(${test_bin_name} ${link_lib} ${OpenCV_LIBS} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(${test_bin_name} PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(${test_bin_name} PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(${test_bin_name} PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(${test_bin_name} PUBLIC ./include)

enable_testing()
