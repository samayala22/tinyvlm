set_project("vlm")
set_version("0.1.0")

add_rules("mode.debug", "mode.release", "mode.releasedbg", "mode.asan")

-- set_toolchains("clang")
-- set_toolset("cxx", "clang")
-- set_policy("build.sanitizer.address", true)
set_policy("build.warning", true)
-- set_policy("build.optimization.lto")
set_warnings("all")
set_languages("c++20", "c99")
set_runtimes("MD")

add_vectorexts("avx2", "fma")

-- if is_mode("release", "releasedbg") then
--     add_cxxflags("/openmp:llvm", {tools = "msvc"})
-- end

if is_mode("debug", "releasedbg") then
    add_defines("TBB_USE_THREADING_TOOLS")
end

if is_plat("windows") then
    add_defines("WIN32")
end

add_includedirs("headeronly", {public = true})

includes("vlm/xmake.lua")
