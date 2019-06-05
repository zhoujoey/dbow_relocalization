cmake_minimum_required( VERSION 2.8 )
project( loop_closure )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
#set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS ${PROJECT_SOURCE_DIR}/DBoW3/lib/libDBoW3.so )

add_executable( feature_training src/feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/DBoW3/lib/libDBoW3.so)
# ${DBoW3_LIBS} )

add_executable( gen_database src/gen_database.cpp )
target_link_libraries( gen_database ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( search_id src/search_id.cpp )
target_link_libraries( search_id ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( gen_vocab src/gen_vocab_large.cpp )
target_link_libraries( gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )
