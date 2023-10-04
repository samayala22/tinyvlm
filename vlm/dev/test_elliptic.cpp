#include "vlm.hpp"
#include "parser.hpp"
#include "vlm_types.hpp"
#include "vlm_solver.hpp"

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
    std::string filename_mesh = parser.get<std::string>("mesh");

    tiny::Config cfg(filename_config);

    vlm::VLM vlm(cfg);

    vlm.mesh.io_read(filename_mesh);
    vlm.init();
    vlm::Solver solver(vlm.mesh, vlm.data, cfg);
    float alpha = 0.1f * vlm::PI_f / 180.0f;
    solver.run(alpha);
        
    return 0;
}