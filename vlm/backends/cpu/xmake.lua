add_requires("openblas_custom")
add_requires("taskflow_custom")

target("backend-cpu")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("taskflow_custom")
    add_packages("openblas_custom", { public = true })

    add_rules("utils.ispc", {header_extension = "_ispc.h"})
    set_values("ispc.flags", {"--target=host", "-O1"})
    add_files("src/*.ispc")

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})
