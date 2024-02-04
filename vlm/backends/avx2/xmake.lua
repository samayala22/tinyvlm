add_requires("tbb")
add_requires("openblas")

target("backend-avx2")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("tbb")
    add_defines("HAVE_LAPACK_CONFIG_H")
    add_packages("openblas")

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})

