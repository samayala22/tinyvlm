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

#include <type_traits> // std::is_same, std::is_integral, std::is_floating_point, std::disjunction
#include <unordered_map> // std::unordered_map
#include <vector> // std::vector
#include <array> // std::array
#include <string> // std::string
#include <sstream> // std::istringstream
#include <fstream> // std::ifstream, std::ofstream
#include <filesystem> // std::filesystem
#include <stdexcept> // std::runtime_error
#include <initializer_list> // std::initializer_list
#include <iostream> // dbg

namespace tiny {

constexpr auto err_missing_section = [](const std::string &s) {throw std::runtime_error("Section [" + s + "] not found");};
constexpr auto err_missing_setting = [](const std::string &s) {throw std::runtime_error("Setting (" + s + ") not found");};
constexpr auto err_missing_config = [](const std::string &s) {throw std::runtime_error("Config file: " + s + " not found");};
constexpr auto err_missing_closing = [](const std::string &s) {throw std::runtime_error("Closing character is missing on line: " + s);};
constexpr auto err_failed_read = [](const std::string &s) {throw std::runtime_error("Failed to read :" + s);};
constexpr auto err_failed_write = [](const std::string &s) {throw std::runtime_error("Failed to write :" + s);};

// Type traits
template<typename T>
struct is_base_type : std::disjunction<
    std::is_integral<T>, 
    std::is_floating_point<T>,
    std::is_same<T, bool>,
    std::is_same<std::decay_t<T>, std::string>
> {};

// Trait for std::vector of base types
template<typename T>
struct is_vector_of_base : std::false_type {};

template<typename T>
struct is_vector_of_base<std::vector<T>> : std::true_type {};

// Trait for std::array of base types
template<typename T>
struct is_array_of_base : std::false_type {};

template<typename T, std::size_t N>
struct is_array_of_base<std::array<T, N>> : std::true_type {};

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
inline T convert_value(const std::string& s) {
    // static_assert(std::is_arithmetic_v<T>, "Invalid type for convert_value. Must be an arithmetic type.");
    T value;
    std::istringstream stream(s);
    stream >> value;
    if (stream.fail()) throw 
    std::runtime_error(
        "Failed to convert value: " + s + " to type: " + data_type_string<T>()
    );
    return value;
}

template<>
inline std::string convert_value(const std::string& s) {
    return s;
}

template<>
inline bool convert_value(const std::string& s) {
    if      (s == "1" || s == "true"  || s == "yes" || s == "on" ) return true;
    else if (s == "0" || s == "false" || s == "no"  || s == "off") return false;
    else throw std::runtime_error("'" + s + "' is not a valid boolean value.");
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

inline bool extract_depth0_token(std::string_view& stream, std::string_view& token, const char delimiter, const char marker_open, const char marker_close) {
    int depth = 0;
    // Loop through each character in the stream
    for(auto it = stream.begin(); it != stream.end(); ++it) {
        if (*it == marker_open) {
            depth++;
        } else if (*it == marker_close) {
            if (depth > 0) {
                depth--;
                if (depth == 0) {
                    // Extract last possible token
                    token = stream.substr(0, it - stream.begin() + 1);
                    stream.remove_prefix(it - stream.begin() + 1);
                    return true;
                }
            } else {
                // Mismatched delimiters
                return false;
            }
        } else if (*it == delimiter && depth == 0) {
            // Extract the token from the stream
            token = stream.substr(0, it - stream.begin());
            stream.remove_prefix(it - stream.begin() + 1);
            return true;
        }
    }
    return false; // no match
}

// Base template for Converter
template<typename T, typename Enable = void>
struct Converter;

// Specialization for base types
template<typename T>
struct Converter<T, std::enable_if_t<is_base_type<T>::value>> {
    // enable only for float or integer types
    static T convert(const std::string& s) {
        return convert_value<T>(s);
    }
};

// Specialization for std::vector of base types
template<typename T>
struct Converter<std::vector<T>, std::enable_if_t<is_vector_of_base<std::vector<T>>::value>> {
    static std::vector<T> convert(const std::string& s) {
        if (s.front() == '[' && s.back() == ']') {
            if (s.size() > 2) {
                std::vector<T> result;
                std::istringstream stream(s.substr(1, s.size() - 2));
                std::string token;
                while (std::getline(stream, token, ',')) result.push_back(Converter<T>::convert(clean_str(token)));
                return result;
            } else {
                return {};
            }
        } else {
            throw std::runtime_error("Vector must be enclosed in square brackets. String: " + s);
        }
    }
};

template<typename T, std::size_t N>
struct Converter<std::array<T, N>, std::enable_if_t<is_array_of_base<std::array<T, N>>::value>> {
    static std::array<T, N> convert(const std::string& s) {
        if (s.front() == '[' && s.back() == ']') {
            if (s.size() > 2) {
                std::array<T, N> result;
                std::istringstream stream(s.substr(1, s.size() - 2));
                std::string token;
                for (std::size_t i = 0; i < N; i++) {
                    if (std::getline(stream, token, ',')) {
                        std::cout << s << " " << token << std::endl;
                        result[i] = Converter<T>::convert(clean_str(token));
                    }
                    else throw std::runtime_error("Array size mismatch. Expected: " + std::to_string(N) + " elements. String: " + s);
                }
                return result;
            } else {
                return {};
            }
        } else {
            throw std::runtime_error("Array must be enclosed in square brackets. String: " + s);
        }
    }
};

template<typename T>
class SectionEntries {
    public:
    std::unordered_map<std::string, T> map_;

    bool has(const std::string &key) const { return map_.find(key) != map_.end(); }

    T& section(const std::string &key) {
        return section_impl(this, key);
    }

    const T& section(const std::string &key) const {
        return section_impl(this, key);
    }

    private:

    template <typename Self>
    static auto& section_impl(Self* self, const std::string& key) {
        auto elem = self->map_.find(key);
        if (elem == self->map_.end()) err_missing_section(key);
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
        return Converter<T>::convert(elem->second);
    }

    template<typename T>
    T get(const std::string& key, const T default_val) const {
        if (has(key)) return get<T>(key);
        else return default_val;
    }

    void insert(const std::initializer_list<std::pair<std::string, std::string>>& pairs) {
        for (const auto& pair : pairs) map_.insert(pair);
    }
};

class Config {
    public:
        SectionEntries<KeyEntries> config;
        SectionEntries<std::vector<KeyEntries>> config_vec;
        std::vector<std::string> sections;

        auto& operator()() {return config;}
        const auto& operator()() const {return config;}

        auto& vector() {return config_vec;}
        const auto& vector() const {return config_vec;}

        void create_section(const std::string& section) {
            config.map_[section];
            config_vec.map_[section];
            sections.push_back(section);
        }

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



inline static void extract(KeyEntries &map, const std::string& line, const size_t sep, const size_t end, const int line_nb) {
    if (sep == std::string::npos) return;
    if (end == std::string::npos) err_missing_closing(std::to_string(line_nb));
    map.map_[clean_str(line.substr(0, sep))] = clean_str(line.substr(sep + 1, end - sep - 1));
}

inline void Config::read_file(std::ifstream& f) {
    std::string line;
    std::string section;
    const std::string whitespaces (" \t\f\v\n\r");
    int line_nb = 0;
    while (std::getline(f, line)) {
        line_nb++;
        // strip the line with everything after the # character
        if (auto comment = line.find('#'); comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        if (auto first_bracket = line.find('<'); first_bracket != std::string::npos) {
            auto second_bracket = line.find('>');
            if (second_bracket == std::string::npos) err_missing_closing(std::to_string(line_nb));
            section = clean_str(line.substr(first_bracket + 1, second_bracket - first_bracket - 1));
            config.map_[section];
            sections.push_back(section);
        } else if (auto first_abracket = line.find('{'); first_abracket != std::string::npos) {
            KeyEntries entry;
            if (first_abracket == line.find_last_not_of(whitespaces)) {
                while (std::getline(f, line) && line.find('}') == std::string::npos) {
                    line_nb++;
                    extract(entry, line, line.find('='), line.find(','), line_nb);
                }
            } else if (line.find('}') != std::string::npos) {
                line = line.substr(first_abracket + 1);
                auto comma = line.find(',');
                while (comma != std::string::npos) {
                    extract(entry, line, line.find('='), comma, line_nb);
                    line = line.substr(comma + 1, std::string::npos);
                    comma = line.find(',');
                }
                extract(entry, line, line.find('='), line.find('}'), line_nb);
            }
            config_vec.map_[section].push_back(std::move(entry));
        } else {
            extract(config.map_[section], line, line.find('='), line.length(), line_nb);
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