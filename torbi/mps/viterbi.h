#pragma once

#include <fstream>
#include <sstream>
#include <string>

inline std::string loadShaderSource(const char* filename) {
    // __FILE__ expands to the full path of this source/header file
    std::string filePath = __FILE__;

    // strip off the file name, leave directory
    std::size_t pos = filePath.find_last_of("/\\");
    std::string dir = (pos == std::string::npos) ? "" : filePath.substr(0, pos + 1);

    // build full path
    std::string fullPath = dir + filename;

    std::ifstream file(fullPath.c_str());
    if (!file.is_open()) {
        throw std::runtime_error("Could not open shader file: " + fullPath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string viterbi_mps_lib = loadShaderSource("viterbi.metal");
