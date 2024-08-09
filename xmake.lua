set_project("vlm")
set_version("0.1.0")
set_xmakever("2.8.6") -- xmake test support

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.asan")

-- set_toolchains("cuda")

-- set_toolset("cxx", "clang")
-- set_policy("build.sanitizer.address", true) -- use xmake f --policies=build.sanitizer.address
set_policy("build.warning", true)
set_policy("build.cuda.devlink", true) -- magic
set_policy("run.autobuild", true)
-- set_policy("build.optimization.lto")

set_warnings("all")
set_languages("c++17", "c99")
set_runtimes("MT") -- msvc runtime library (MD/MT/MDd/MTd)

-- Define backends and helper functions
backends = {"cuda", "cpu"}
backend_includes = function(name) return "vlm/backends/" .. name .. "/xmake.lua" end
backend_defines  = function(name) return "VLM_" .. name:upper() end
backend_deps     = function(name) return "backend-" .. name end
backend_option   = function(name) return "build-" .. name end

-- Headeronly libraries
add_includedirs("headeronly", {public = true}) -- must be set before options

-- Create compilation options and include backends accordingly
for _,name in ipairs(backends) do
    -- Create option
    option(backend_option(name))
        set_default(name == "cpu") -- set default to true for cpu, false otherwise
        set_showmenu(true)
    option_end()
    -- Add option includes
    if has_config(backend_option(name)) then
        includes(backend_includes(name))
    end
end

includes("packages/*.lua")
includes("vlm/xmake.lua") -- library and main driver

-- Create tests
for _, file in ipairs(os.files("tests/*.cpp")) do
    local name = path.basename(file)
    target(name)
        set_kind("binary")
        set_default(false)
        add_rpathdirs("$ORIGIN")
        add_deps("libvlm")
        add_files("tests/" .. name .. ".cpp")
        add_tests("default")
end
