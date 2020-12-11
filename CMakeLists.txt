cmake_minimum_required(VERSION 3.0)
project(NeuralNetwork)
add_executable(NeuralNetwork main.cpp backpropagation.h backpropagation.cpp)
install(TARGETS NeuralNetwork DESTINATION bin)