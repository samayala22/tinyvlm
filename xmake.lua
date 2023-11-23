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
set_runtimes("MD")

if is_mode("debug", "releasedbg") then
    add_defines("TBB_USE_THREADING_TOOLS")
end

if is_plat("windows") then
    add_defines("WIN32")
end

add_includedirs("headeronly", {public = true}) -- must be set before options

option("avx2")
    set_default(true)
    set_showmenu(true)
    add_defines("VLM_AVX2")
    -- add_deps("backend-avx2")
option_end()

option("cuda")
    set_default(false)
    set_showmenu(true)
    add_defines("VLM_CUDA")
    -- add_deps("backend-cuda")
option_end()

includes("**/xmake.lua")
