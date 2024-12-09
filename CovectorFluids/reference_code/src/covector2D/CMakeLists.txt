find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

include_directories(${TBB_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
add_executable(Covector2D main.cpp CovectorSolver2D.cpp ../utils/writeBMP.cpp)
target_link_libraries(Covector2D ${TBB_LIBRARIES})
target_link_libraries(Covector2D ${Boost_LIBRARIES})