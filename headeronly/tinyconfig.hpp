/*
MIT License

Copyright (c) 2023 Samuel Ayala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Version: 0.2
// Last updated: 01/09/2023

#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace tiny {

constexpr auto err_missing_section = [](const std::string &s) {throw std::runtime_error("Section [" + s + "] not found");};
constexpr auto err_missing_setting = [](const std::string &s) {throw std::runtime_error("Setting (" + s + ") not found");};
constexpr auto err_missing_config = [](const std::string &s) {throw std::runtime_error("Config file: " + s + " not found");};
constexpr auto err_missing_closing = [](const std::string &s) {throw std::runtime_error("Closing character is missing on line: " + s);};
constexpr auto err_failed_read = [](const std::string &s) {throw std::runtime_error("Failed to read :" + s);};
constexpr auto err_failed_write = [](const std::string &s) {throw std::runtime_error("Failed to write :" + s);};

template <typename T> inline std::string data_type_string() {
    std::string typestr;

    if constexpr (std::numeric_limits<T>::is_integer) {
        typestr = std::numeric_limits<T>::is_signed ? "Int" : "UInt";
    } else {
        typestr = "Float";
    }
    typestr.append(std::to_string(sizeof(T) * 8));
    return typestr;
}

template<typename T>
inline T convert_value(std::string s) {
    //static_assert(std::is_arithmetic_v<T>, "Invalid type for convert_value. Must be an arithmetic type.");
    T value;
    std::istringstream stream(s);
    stream >> value;
    if (stream.fail()) throw std::runtime_error("Failed to convert value: " + s + " to type: " + data_type_string<T>());
    return value;
}

template<>
inline std::string convert_value(std::string s) {
    return s;
}

template<>
inline bool convert_value(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    static const std::unordered_map<std::string, bool> s2b{
        {"1", true},  {"true", true},   {"yes", true}, {"on", true},
        {"0", false}, {"false", false}, {"no", false}, {"off", false},
    };
    auto const val = s2b.find(s);
    if (val == s2b.end()) {
        throw std::runtime_error("'" + s + "' is not a valid boolean value.");
    }
    return val->second;
}

template<typename T>
inline std::vector<T> convert_vector(std::string s) {
    if (s.front() == '[' && s.back() == ']') {
        if (s.size() > 2) {
            std::vector<T> result;
            s = s.substr(1, s.size() - 2);
            std::istringstream stream(s);
            std::string token;
            while (std::getline(stream, token, ',')) {
                result.push_back(convert_value<T>(token));
            }
            return result;
        } else {
            return {};
        }
    } else {
        throw std::runtime_error("Vector must be enclosed in square brackets. String: " + s);
    }
}

template<typename T>
struct BaseEntries {
    std::unordered_map<std::string, T> map_;
    bool has(const std::string &key) const { return map_.find(key) != map_.end(); }
    // not const as it can be used to modify the map
    T& section(const std::string &key, bool create = false) {
        if (create) return map_[key];
        auto elem = map_.find(key);
        if (elem == map_.end()) err_missing_section(key);
        return elem->second;
    }
};

struct KeyEntries {
    std::unordered_map<std::string, std::string> map_;
    bool has(const std::string &key) const { return map_.find(key) != map_.end(); }
    
    template<typename T>
    T get(const std::string &key) const {
        auto elem = map_.find(key);
        if (elem == map_.end()) err_missing_setting(key);
        return convert_value<T>(elem->second);
    }

    template<typename T>
    T get(const std::string& key, const T default_val) const {
        if (has(key)) return get<T>(key);
        else return default_val;
    }

    template<typename T>
    std::vector<T> get_vector(const std::string &key) const {
        auto elem = map_.find(key);
        if (elem == map_.end()) err_missing_setting(key);
        return convert_vector<T>(elem->second);
    }

    template<typename T>
    std::vector<T> get_vector(const std::string& key, const std::vector<T> default_val) const {
        if (has(key)) return get_vector<T>(key);
        else return default_val;
    }
};

class Config {
    public:
        BaseEntries<KeyEntries> config;
        BaseEntries<std::vector<KeyEntries>> config_vec;
        std::vector<std::string> sections;

        auto& operator()() {return config;}
        auto& vector() {return config_vec;}
        const auto& operator()() const {return config;}
        const auto& vector() const {return config_vec;}

        bool read(const std::string& ifilename);
        bool write(const std::string& ofilename);
        Config() = default;
        Config(const std::string& ifilename) {read(ifilename);}

    private:
        void read_file(std::ifstream &ifile);
        void write_file(std::ofstream& ofile);
};

inline bool Config::read(const std::string& ifilename) {
    std::filesystem::path path(ifilename);
    if (!std::filesystem::exists(path)) err_missing_config(ifilename);

    std::ifstream f(path);
	if (f.is_open()) {
        read_file(f);
        f.close();
	} else return false;
    return !f.bad();
}

inline bool Config::write(const std::string& ofilename) {
    std::filesystem::path path(ofilename);
    // if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path);
	if (f.is_open()) {
        write_file(f);
        f.close();
	} else return false;
    return !f.bad();
}

inline static std::string clean_str(const std::string& str) {
    static const std::string whitespaces(" \t\f\v\n\r");
    std::size_t start = str.find_first_not_of(whitespaces);
    std::size_t end = str.find_last_not_of(whitespaces);

    if (start == std::string::npos) {
        return "";
    }
    return str.substr(start, end - start + 1);
}

inline static void extract(KeyEntries &map, const std::string& line, const size_t sep, const size_t end, const int line_nb) {
    if (sep == std::string::npos) return;
    if (end == std::string::npos) err_missing_closing(std::to_string(line_nb));
    map.map_[clean_str(line.substr(0, sep))] = clean_str(line.substr(sep + 1, end - sep - 1));
}

inline void Config::read_file(std::ifstream &f) {
    std::string line;
    std::string section;
    const std::string whitespaces (" \t\f\v\n\r");
    int line_nb = 0;
    while (std::getline(f, line)) {
        line_nb++;
        // strip the line with everything after the # character
        if (auto comment = line.find("#"); comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        if (auto first_bracket = line.find("<"); first_bracket != std::string::npos) {
            auto second_bracket = line.find(">");
            if (second_bracket == std::string::npos) err_missing_closing(std::to_string(line_nb));
            section = clean_str(line.substr(first_bracket + 1, second_bracket - first_bracket - 1));
            config.map_[section];
            sections.push_back(section);
        } else if (auto first_abracket = line.find("{"); first_abracket != std::string::npos) {
            KeyEntries entry;
            if (first_abracket == line.find_last_not_of(whitespaces)) {
                while (std::getline(f, line) && line.find("}") == std::string::npos) {
                    line_nb++;
                    extract(entry, line, line.find("="), line.find(","), line_nb);
                }
            } else if (line.find("}") != std::string::npos) {
                line = line.substr(first_abracket + 1);
                auto comma = line.find(",");
                while (comma != std::string::npos) {
                    extract(entry, line, line.find("="), comma, line_nb);
                    line = line.substr(comma + 1, std::string::npos);
                    comma = line.find(",");
                }
                extract(entry, line, line.find("="), line.find("}"), line_nb);
            }
            config_vec.map_[section].push_back(std::move(entry));
        } else {
            extract(config.map_[section], line, line.find("="), line.length(), line_nb);
        }
    }
}

inline void Config::write_file(std::ofstream &f) {
    for (const auto &section : sections) {
        f << "<" << section << "> \n";
        for (const auto& [setting, value] : config.map_[section].map_) {
            f << setting << " = " << value << "\n";
        }
        // loop over vector elements (multiple KeyEntries)
        for (const auto &entries : config_vec.map_[section]) {
            if (entries.map_.size() > 1) {
                f << "<\n";
                for (const auto& [setting, value] : entries.map_) {
                    f << "\t" << setting << " = " << value << ",\n";
                }
                f << ">\n";
            } else if (entries.map_.size() == 1){
                f << "{" << entries.map_.begin()->first << " = " << entries.map_.begin()->second << ">\n";
            }
        }
        f << "\n";
    }
}
} // namespace tiny