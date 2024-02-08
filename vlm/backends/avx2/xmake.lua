add_requires("openblas_custom")
add_requires("taskflow_custom")

target("backend-avx2")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("taskflow_custom")
    add_defines("HAVE_LAPACK_CONFIG_H")
    add_packages("openblas_custom", { public = true })

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})
