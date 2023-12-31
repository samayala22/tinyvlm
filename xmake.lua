set_project("vlm")
set_version("0.1.0")
set_xmakever("2.8.3") -- autobuild support

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.asan")

-- set_toolchains("cuda")

-- set_toolset("cxx", "clang")
-- set_policy("build.sanitizer.address", true)
set_policy("build.warning", true)
set_policy("build.cuda.devlink", true) -- magic
set_policy("run.autobuild", true)
-- set_policy("build.optimization.lto")

set_warnings("all")
set_languages("c++20", "c99")
set_runtimes("MD") -- msvc runtime library (MD/MT/MDd/MTd)

-- TBB macro for profiling parallel objects
if is_mode("debug", "releasedbg") then
    add_defines("TBB_USE_THREADING_TOOLS")
end

-- Define backends and helper functions
backends = {"cuda", "avx2"}
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
        set_default(name == "avx2") -- set default to true for avx2, false otherwise
        set_showmenu(true)
    option_end()
    -- Add option includes
    if has_config(backend_option(name)) then
        includes(backend_includes(name))
    end
end

includes("vlm/xmake.lua") -- library and main driver
