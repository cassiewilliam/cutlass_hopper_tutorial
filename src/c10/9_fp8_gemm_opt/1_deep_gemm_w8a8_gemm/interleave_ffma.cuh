#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace fs = std::filesystem;

namespace deep_gemm::jit
{

std::string run_cuobjdump(const std::string& file_path)
{
    // Check CUDA_HOME
    char const* CUDA_HOME = std::getenv("CUDA_HOME");
    std::string command = std::string(CUDA_HOME) + "/bin/cuobjdump -sass " + file_path;
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    int return_code = pclose(pipe);
    assert(return_code == 0);
    return result;
}

std::vector<std::pair<std::string, std::vector<std::string>>> extract_ffma(const std::string& sass) {
    std::vector<std::pair<std::string, std::vector<std::string>>> collected;
    std::vector<std::string> current;
    std::string arch_name = "N/A", func_name = "N/A";
    bool skip_next_line = false;
    std::istringstream stream(sass);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("code for") != std::string::npos) {
            arch_name = line.substr(line.find("code for") + 9);
        } else if (line.find("Function :") != std::string::npos) {
            func_name = line.substr(line.find("Function :") + 10);
        } else if (line.find("FFMA") != std::string::npos) {
            current.push_back(line);
            skip_next_line = true;
        } else if (skip_next_line) {
            current.push_back(line);
            skip_next_line = false;
        } else {
            if (current.size() >= 16 && current.size() % 2 == 0) {
                collected.emplace_back(arch_name + "::" + func_name, current);
            }
            current.clear();
        }
    }
    return collected;
}


// Function to extract hexadecimal number from the line
uint64_t extract_hex_from_line(const std::string& line) {
    // Regular expression to match hex numbers in comments (e.g., /* 0x1234abcd */)
    std::regex hex_pattern(R"(/\*\s*(0x[0-9a-fA-F]+)\s*\*/)");

    std::smatch match;
    if (std::regex_search(line, match, hex_pattern)) {
        // Convert the matched hex string to an integer
        return std::stoull(match.str(1), nullptr, 16);
    } else {
        // If no match is found, assert failure
        assert(false && "No hex number found in the comment");
        return 0; // To satisfy the compiler, though this will not be reached
    }
}

std::vector<std::string> parse_registers(const std::string& line)
{
    std::vector<std::string> registers;

    // Step 1: Remove comments (everything inside /* */)
    std::string modified_line = std::regex_replace(line, std::regex(R"(/\*.*?\*/)", ""), "");

    // Step 2: Remove any semicolon
    modified_line.erase(std::remove(modified_line.begin(), modified_line.end(), ';'), modified_line.end());

    // Step 3: Split the line by commas
    std::istringstream stream(modified_line);
    std::string token;
    while (std::getline(stream, token, ',')) {
        // Step 4: Trim leading and trailing spaces from the token
        token.erase(0, token.find_first_not_of(' '));  // Trim leading spaces
        token.erase(token.find_last_not_of(' ') + 1); // Trim trailing spaces

        // Step 5: Split token by spaces and check for register names
        std::istringstream token_stream(token);
        std::string word;
        while (token_stream >> word) {
            if (word.rfind("R", 0) == 0) { // Check if word starts with 'R'
                std::string reg = word.substr(0, word.find('.'));  // Remove the . if exists
                registers.push_back(reg);
            }
        }
    }

    return registers;
}


// Function to validate the memory region
bool validate(const char* m, size_t offset, const std::vector<std::string>& le_bytes, size_t num_lines) {
    // Step 1: Check if the length of le_bytes matches half of num_lines
    assert(le_bytes.size() == num_lines / 2);

    // Step 2: Check if the memory at the offset matches the first byte sequence
    if (std::memcmp(m + offset, le_bytes[0].data(), 16) != 0) {
        return false;
    }

    // Step 3: Check if the subsequent memory segments match the rest of le_bytes
    for (size_t i = 1; i < num_lines / 2; i++) {
        if (std::memcmp(m + offset + i * 16, le_bytes[i].data(), 16) != 0) {
            return false;
        }
    }

    return true;
}

void print_reg_reuse(const std::string& name, size_t num_changed, const std::vector<size_t>& reused_list) {
    const char* print_reg_reuse_env = std::getenv("DG_PRINT_REG_REUSE");

    // Check if the environment variable is set (not null)
    if (print_reg_reuse_env != nullptr) {
        std::cout << " > segment `" << name << "` new reused list (" << num_changed 
                  << " changed): ";
        
        // Print the reused list
        for (size_t reg : reused_list) {
            std::cout << reg << " ";
        }
        std::cout << std::endl;
    }
}

void filter_offsets(std::vector<size_t>& offsets, const char* m, const std::vector<std::string>& le_bytes, size_t num_lines) {
    // Create a new vector to hold the filtered offsets
    std::vector<size_t> filtered_offsets;

    // Use std::copy_if to filter offsets based on the validate function
    std::copy_if(offsets.begin(), offsets.end(), std::back_inserter(filtered_offsets),
                 [m, &le_bytes, num_lines](size_t offset) {
                     return validate(m, offset, le_bytes, num_lines); // Call validate for each offset
                 });

    // Replace the original offsets with the filtered offsets
    offsets = std::move(filtered_offsets);
}

void modify_segment(char* m, const std::string& name, const std::vector<std::string>& ffma_lines) {
    size_t num_lines = (ffma_lines.size() * 9 / 16) / 2 * 2;
    assert(num_lines % 2 == 0);

    std::vector<std::string> le_bytes, new_le_bytes;
    std::vector<size_t> reused_list;
    std::unordered_set<std::string> dst_reg_set;
    bool last_reused = false;
    std::string last_dst_reg;
    size_t num_changed = 0;
    
    for (size_t i = 0; i < num_lines / 2; i++)
    {
        auto registers = parse_registers(ffma_lines[i * 2]);
        std::string dst_reg = registers[registers.size() - 2];  // Access the second-to-last element

        uint64_t low_hex = extract_hex_from_line(ffma_lines[i * 2]);
        uint64_t high_hex = extract_hex_from_line(ffma_lines[i * 2 + 1]);

        bool reused = (high_hex & 0x0800000000000000) != 0;
        if (reused) {
            bool is_first_occurred = dst_reg_set.find(dst_reg) == dst_reg_set.end();
            if (is_first_occurred || (last_reused && dst_reg == last_dst_reg)) {
                assert(high_hex & 0x0800200000000000);
                high_hex ^= 0x0800200000000000;
                reused = false;
                num_changed++;
            } else {
                reused_list.push_back(i);
            }
        }
        dst_reg_set.insert(dst_reg);
        
        le_bytes.push_back(std::string(reinterpret_cast<char*>(&low_hex), 8) + std::string(reinterpret_cast<char*>(&high_hex), 8));
        new_le_bytes.push_back(std::string(reinterpret_cast<char*>(&low_hex), 8) + std::string(reinterpret_cast<char*>(&high_hex), 8));
        
        last_reused = reused;
        last_dst_reg = dst_reg;
    }

    print_reg_reuse(name, num_changed, reused_list);

    std::vector<size_t> offsets;
    size_t offset = std::search(m, m + strlen(m), le_bytes[0].begin(), le_bytes[0].end()) - m;
    while (offset < strlen(m)) {
        offsets.push_back(offset);
        offset = std::search(m + offset + 1, m + strlen(m), le_bytes[0].begin(), le_bytes[0].end()) - m;
    }

    filter_offsets(offsets, m, le_bytes, num_lines);

    for (size_t offset : offsets) {
        for (size_t i = 0; i < num_lines / 2; i++) {
            std::memcpy(m + offset + i * 16, new_le_bytes[i].data(), 16);
        }
    }
}

void interleave_ffma_process(const std::string& path) {
    std::string output = run_cuobjdump(path);
    auto segments = extract_ffma(output);

    int fd = open(path.c_str(), O_RDWR);
    struct stat sb;
    fstat(fd, &sb);
    void* file_mem = mmap(nullptr, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(file_mem != MAP_FAILED);
    char* mem = static_cast<char*>(file_mem);

    for (const auto& segment : segments) {
        modify_segment(mem, segment.first, segment.second);
    }

    munmap(file_mem, sb.st_size);
    close(fd);
}
    
}