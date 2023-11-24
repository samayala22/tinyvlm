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

option("build-avx2")
    set_default(true)
    set_showmenu(true)
option_end()

option("build-cuda")
    set_default(false)
    set_showmenu(true)
option_end()

includes("**/xmake.lua")
