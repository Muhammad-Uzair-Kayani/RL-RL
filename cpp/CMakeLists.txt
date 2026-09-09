cmake_minimum_required(VERSION 3.18)
project(RLTeamSportsEnv CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(teamsports_rl MODULE rl_env.cpp)
target_link_libraries(teamsports_rl PRIVATE pybind11::module)
set_target_properties(teamsports_rl PROPERTIES PREFIX ""
                                               SUFFIX ".pyd")
