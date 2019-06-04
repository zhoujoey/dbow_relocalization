cmake_minimum_required( VERSION 2.8 )
project(voc2_demo)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
#set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
#set( DBoW2_LIBS "${PROJECT_SOURCE_DIR}/DBoW2/lib/libDBoW2.so" )


include_directories(src/)

add_executable( feature_training src/feature_training.cpp src/DBOW.h)
target_link_libraries( feature_training ${OpenCV_LIBS} 
${PROJECT_SOURCE_DIR}/DBoW2/lib/libDBoW2.so
)


add_executable( loop_closure src/loop_closure.cpp src/DBOW.h)
target_link_libraries( loop_closure ${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/DBoW2/lib/libDBoW2.so
)

add_executable( search_id src/search_id.cpp src/DBOW.h)
target_link_libraries( search_id ${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/DBoW2/lib/libDBoW2.so
)
