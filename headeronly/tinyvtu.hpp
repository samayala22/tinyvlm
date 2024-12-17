// Heavily modified and simplified version of https://github.com/phmkopp/vtu11
/*
BSD 3-Clause License

Copyright (c)
2019, Philipp Kopp, Philipp Bucher, Technical University of Munich
2024, Samuel Ayala, Polytechnique Montreal
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <tuple>

namespace tiny {

using StringStringMap = std::map<std::string, std::string>;

template<typename T>
class VtuDataAccessor {
public:
    virtual ~VtuDataAccessor() = default;
    virtual T& operator[](std::int64_t i) const = 0;
    virtual std::int64_t size() const = 0;
    virtual std::int32_t components() const = 0;
};

template<typename VertT, typename IdxT, typename CellT>
struct VtuMesh {
    std::unique_ptr<VtuDataAccessor<VertT>> vertices;
    std::unique_ptr<VtuDataAccessor<IdxT>> connectivity;
    std::unique_ptr<VtuDataAccessor<IdxT>> offsets;
    std::unique_ptr<VtuDataAccessor<CellT>> types;

    std::int64_t number_of_points() const { return vertices->size() / vertices->components(); }
    std::int64_t number_of_cells() const { return types->size(); }
};

template<typename T>
using VtuDataSet = std::vector<std::tuple<std::string, std::unique_ptr<VtuDataAccessor<T>>>>;

namespace detail {

constexpr size_t BUFFER_SIZE = 1u << 15;

inline std::string endianness() {
    int i = 0x0001;
    if (*reinterpret_cast<char *>(&i) != 0) {
        return "LittleEndian";
    }
    return "BigEndian";
}

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

inline void write_tag(std::ostream &output, const std::string &name, bool close,
                      bool single, const StringStringMap &attributes = {}) {
    if (close) {
        output << "</";
    } else {
        output << "<";
    }
    output << name;
    for (const auto &attribute : attributes) {
        output << " " << attribute.first << "=\"" << attribute.second << "\"";
    }
    if (single) {
        output << "/>\n";
    } else {
        output << ">\n";
    }
}

inline void write_empty_tag(std::ostream &output, const std::string &name,
                            const StringStringMap &attributes = {}) {
    write_tag(output, name, false, true, attributes);
}

class ScopedXmlTag final {
public:
    ScopedXmlTag(std::ostream &output, const std::string &name,
                 const StringStringMap &attributes={});

    ~ScopedXmlTag();

private:
    std::ostream &m_output;
    const std::string m_name;
};

inline ScopedXmlTag::ScopedXmlTag(std::ostream &output, const std::string &name,
                                  const StringStringMap &attributes)
    : m_output(output), m_name(name) {
    write_tag(output, name, false, false, attributes);
}

inline ScopedXmlTag::~ScopedXmlTag() {
    write_tag(m_output, m_name, true, false, {});
}

struct DummyWriter {
    void add_header_attributes(StringStringMap &) {}
    void add_data_attributes(StringStringMap &) {}
};

struct AsciiWriter {
    template <typename T>
    void write_data(std::ostream &output, const std::unique_ptr<VtuDataAccessor<T>>& data);
    void write_appended(std::ostream &output);
    void add_header_attributes(StringStringMap &attributes);
    void add_data_attributes(StringStringMap &attributes);

    StringStringMap appended_attributes();
};

template <typename T>
inline void AsciiWriter::write_data(std::ostream &output, const std::unique_ptr<VtuDataAccessor<T>>& data) {
    for (std::int64_t i = 0; i < data->size(); i+=data->components()) {
        for (std::int32_t j = 0; j < data->components(); j++) {
            output << (*data)[i+j] << " ";
        }
        output << "\n";
    }
}

template <>
inline void AsciiWriter::write_data(std::ostream &output, const std::unique_ptr<VtuDataAccessor<std::int8_t>>& data) {
    for (std::int64_t i = 0; i < data->size(); i+=data->components()) {
        for (std::int32_t j = 0; j < data->components(); j++) {
            output << static_cast<int>((*data)[i+j]) << " ";
        }
        output << "\n";
    }
}

inline void AsciiWriter::write_appended(std::ostream &) {}

inline void AsciiWriter::add_header_attributes(StringStringMap &) {}

inline void AsciiWriter::add_data_attributes(StringStringMap &attributes) {
    attributes["format"] = "ascii";
}

inline StringStringMap AsciiWriter::appended_attributes() { return {}; }

template <typename DataType, typename Writer>
inline StringStringMap write_data_set_header(
    Writer &&writer,
    const std::string &name,
    size_t ncomponents
) {
    StringStringMap attributes = {{"type", data_type_string<DataType>()}};

    if (name != "") {
        attributes["Name"] = name;
    }

    if (ncomponents > 1) {
        attributes["NumberOfComponents"] = std::to_string(ncomponents);
    }

    writer.add_data_attributes(attributes);

    return attributes;
}

template <typename Writer, typename DataT>
inline void write_data_set(
    Writer &&writer,
    std::ostream &output,
    const std::string &name,
    const size_t ncomponents,
    const std::unique_ptr<VtuDataAccessor<DataT>>& data) {
    auto attributes = write_data_set_header<DataT>(writer, name, ncomponents);

    if (attributes["format"] != "appended") {
        const ScopedXmlTag data_array_tag(output, "DataArray", attributes);
        writer.write_data(output, data);
    } else {
        write_empty_tag(output, "DataArray", attributes);
        writer.write_data(output, data);
    }
}

template <typename Writer, typename DataT>
void write_datasets(
    std::ostream &output,
    const VtuDataSet<DataT> &datasets,
    Writer &&writer
) {
    for (const auto &[name, data] : datasets) {
        write_data_set(writer, output, name, data->components(), data);
    }
}

template <typename Writer, typename DataT>
void write_datasets_pvtu_headers(
    std::ostream &output,
    const VtuDataSet<DataT> &datasets,
    Writer &&writer
) {
    for (const auto &[name, data] : datasets) {
        auto attributes = write_data_set_header<DataT>(writer, name, data->components());
        write_empty_tag(output, "PDataArray", attributes);
    }
}

template <typename Writer, typename Content>
inline void write_file(const std::string &filename, const char *type,
                       Writer &&writer, Content &&writeContent) {
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open())
        throw std::runtime_error("Failed to open file " + filename + "\n");

    std::array<char, BUFFER_SIZE> buffer;

    output.rdbuf()->pubsetbuf(buffer.data(),
                              static_cast<std::streamsize>(buffer.size()));

    output << "<?xml version=\"1.0\"?>\n";

    StringStringMap header_attributes{
        {"byte_order", endianness()}, {"type", type}, {"version", "0.1"}};

    writer.add_header_attributes(header_attributes);

    {
        const ScopedXmlTag vtk_file_tag(output, "VTKFile", header_attributes);

        writeContent(output);

    } // VTKFile

    output.close();
}

template <typename Writer, typename VertT, typename IdxT, typename CellT, typename DataT>
void write(
    const std::string &filename,
    const VtuMesh<VertT, IdxT, CellT>& mesh,
    const VtuDataSet<DataT> &point_data,
    const VtuDataSet<DataT> &cell_data,
    Writer &&writer
) {
    write_file(filename, "UnstructuredGrid", writer, [&](std::ostream &output) {
        {
            const ScopedXmlTag unstructured_grid_file_tag(
                output, "UnstructuredGrid", {});
            {
                const ScopedXmlTag piece_tag(
                    output, "Piece",
                    {{"NumberOfPoints",
                      std::to_string(mesh.number_of_points())},
                     {"NumberOfCells", std::to_string(mesh.number_of_cells())}

                    });

                {
                    const ScopedXmlTag point_data_tag(output, "PointData", {});

                    write_datasets(output, point_data, writer);

                } // PointData

                {
                    const ScopedXmlTag cell_data_tag(output, "CellData", {});

                    write_datasets(output, cell_data, writer);

                } // CellData

                {
                    const ScopedXmlTag points_tag(output, "Points", {});

                    write_data_set(writer, output, "", 3, mesh.vertices);

                } // Points

                {
                    const ScopedXmlTag points_tag(output, "Cells", {});

                    write_data_set(writer, output, "connectivity", 1,
                                   mesh.connectivity);
                    write_data_set(writer, output, "offsets", 1, mesh.offsets);
                    write_data_set(writer, output, "types", 1, mesh.types);

                } // Cells
            }     // Piece
        }         // UnstructuredGrid

        auto appended_attributes = writer.appended_attributes();

        if (!appended_attributes.empty()) {
            const ScopedXmlTag appended_data_tag(output, "AppendedData",
                                                 appended_attributes);

            output << "_";

            writer.write_appended(output);

        } // AppendedData
    });   // writeVTUFile
}

} // namespace detail

template<typename VertT, typename IdxT, typename CellT, typename DataT>
inline void write_vtu(
    const std::string &filename,
    const VtuMesh<VertT, IdxT, CellT>& mesh,
    const VtuDataSet<DataT> &point_data,
    const VtuDataSet<DataT> &cell_data,
    const std::string &write_mode = "ascii"
) {
    if (write_mode == "ascii") {
        write(filename, mesh, point_data, cell_data, detail::AsciiWriter{});
    } else {
        throw std::runtime_error("Invalid write mode: \"" + write_mode + "\".");
    }
}

template<typename DataT>
inline void write_pvtu(
    const std::string &path,
    const std::string &baseName,
    const VtuDataSet<DataT> &point_data,
    const VtuDataSet<DataT> &cell_data,
    const size_t numberOfFiles
) {
    auto directory = std::filesystem::path{path} / baseName;
    auto pvtufile = directory / (baseName + ".pvtu");

    // create directory for vtu files if not existing
    if (!std::filesystem::exists(directory)) {
        std::filesystem::create_directories(directory);
    }

    detail::DummyWriter writer;

    write_file(
        pvtufile.string(), "PUnstructuredGrid", writer,
        [&](std::ostream &output) {
            const detail::ScopedXmlTag p_unstructured_grid_file_tag(
                output, "PUnstructuredGrid", {{"GhostLevel", "0"}});

            {
                const detail::ScopedXmlTag p_point_data_tag(output, "PPointData", {});

                write_datasets_pvtu_headers(output, point_data, writer);

            } // PPointData

            {
                const detail::ScopedXmlTag p_cell_data_tag(output, "PCellData", {});

                write_datasets_pvtu_headers(output, cell_data, writer);

            } // PCellData

            {
                const detail::ScopedXmlTag p_points_tag(output, "PPoints", {});
                StringStringMap attributes = {
                    {"type", detail::data_type_string<float>()}, // TODO: make this template param
                    {"NumberOfComponents", "3"}};

                writer.add_data_attributes(attributes);

                detail::write_empty_tag(output, "PDataArray", attributes);

            } // PPoints

            for (size_t i = 0; i < numberOfFiles; ++i) {
                std::string piece_name = baseName + "_" + std::to_string(i) + ".vtu";
                detail::write_empty_tag(output, "Piece", {{"Source", piece_name}});
            }
        }); // writeVTUFile

} // write_pvtu

template<typename T>
inline void write_pvd(
    const std::string& path,
    const std::string& baseName,
    const std::string& extension, // .vtu or .pvtu
    const std::unique_ptr<VtuDataAccessor<T>>& timesteps
) {
    assert(timesteps->components() == 1);
    auto directory = std::filesystem::path{path} / baseName;
    auto filename = directory / (baseName + ".pvd");

    if (!std::filesystem::exists(directory)) { throw std::runtime_error("Dir is supposed to exist"); }

    detail::DummyWriter writer;

    write_file(filename.string(), "Collection", writer, [&](std::ostream &output) {
        const detail::ScopedXmlTag collection_tag(output, "Collection");
        {
            for (std::int64_t i = 0; i < timesteps->size(); i++) {
                std::string step_name = baseName + "_" + std::to_string(i);
                detail::write_empty_tag(output, "DataSet", {
                    {"timestep", std::to_string((*timesteps)[i])},
                    {"group", ""},
                    {"part", ""},
                    {"file", (std::filesystem::path{step_name} / (step_name + extension)).string() }
                });
            }
        } // Collection
    });
} // writePVD

} // namespace tinyvtu