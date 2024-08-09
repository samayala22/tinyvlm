package("taskflow_custom")
    set_kind("library", {headeronly = true})
    set_homepage("https://taskflow.github.io/")
    set_description("A fast C++ header-only library to help you quickly write parallel programs with complex task dependencies")
    set_license("MIT")
    set_urls("https://github.com/samayala22/taskflow.git")

    if is_plat("linux") then
        add_syslinks("pthread")
    end

    on_install("linux", "macosx", "windows", "iphoneos", "android", "cross", "mingw", "bsd", function (package)
        os.cp("taskflow", package:installdir("include"))
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <taskflow/taskflow.hpp>
            #include <taskflow/algorithm/for_each.hpp>
            static void test() {
                tf::Executor executor;
                tf::Taskflow taskflow;
                std::vector<int> range(10);
                std::iota(range.begin(), range.end(), 0);
                taskflow.for_each(range.begin(), range.end(), [&] (int i) {
                    std::printf("for_each on container item: %d\n", i);
                });
                executor.run(taskflow).wait();
            }
        ]]}, {configs = {languages = "c++17"}}))
    end)