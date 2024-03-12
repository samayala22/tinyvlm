#include "vlm.hpp"
#include "parser.hpp"
#include "vlm_executor.hpp"
#include "vlm_types.hpp"
#include "vlm_utils.hpp"

#include "tinyconfig.hpp"

#include <iostream>
#include <cstdio>
#include <algorithm>

using namespace vlm;

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
    parser.add("database",                 // name
               "Viscous database file",  // description
               "-db",                   // shorthand
               false,                   // required argument
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
    std::string filename_database = parser.get<std::string>("database");

    tiny::Config cfg(filename_config);
    cfg.create_section("files");
    cfg().section("files").insert({
        {"mesh", filename_mesh},
        {"database", filename_database}
    });

    try {
        // vlm::Executor::instance(1); // 1 thread
        LinearVLM solver(cfg);
        std::vector<f32> alphas = cfg().section("solver").get<std::vector<f32>>("alphas");
        std::transform(alphas.begin(), alphas.end(), alphas.begin(), 
        [](f32 deg) {
            return deg * PI_f / 180.0;
        });
        
        for (auto alpha : alphas) {
            FlowData flow(alpha, 0.0f, 1.0f, 1.0f);
            auto coeffs = solver.solve(flow);
            std::printf(">>> Alpha: %.1f | CL = %.6f CD = %.6f CMx = %.6f CMy = %.6f CMz = %.6f\n", vlm::to_degrees(flow.alpha), coeffs.cl, coeffs.cd, coeffs.cm.x, coeffs.cm.y, coeffs.cm.z);
        }

        // Pause for memory reading
        // std::cout << "Done ..." << std::endl;
        // std::cin.get();
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}