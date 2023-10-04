#include "vlm.hpp"
#include "parser.hpp"

void cmdl_parser_configure(cmd_line_parser::parser& parser) {
    parser.add("config",                 // name
               "Configuration file",  // description
               "-i",                   // shorthand
               true,                   // required argument
               false                   // is boolean option
    );
    parser.add("mesh",                 // name
               "Mesh file",  // description
               "-m",                   // shorthand
               true,                   // required argument
               false                   // is boolean option
    );
    parser.add("output",                 // name
               "Results file",  // description
               "-o",                   // shorthand
               true,                   // required argument
               false                   // is boolean option
    );
}

int main(int argc, char **argv) {
    cmd_line_parser::parser parser(argc, argv);
    cmdl_parser_configure(parser);

    bool success = parser.parse();
    if (!success) return 1;

    std::string filename_config = parser.get<std::string>("config");
    tiny::Config cfg(filename_config);

    vlm::VLM vlm(cfg);

    vlm.io.filename_mesh = parser.get<std::string>("mesh");
    vlm.io.filename_result = parser.get<std::string>("output");

    try {
        vlm.init();
        vlm.preprocess();
        vlm.solve(cfg);
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}