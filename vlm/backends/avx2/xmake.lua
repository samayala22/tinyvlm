add_requires("tbb")

add_requires("openblas")
add_requires("taskflow")

target("backend-avx2")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("tbb")
    add_packages("taskflow")
    add_defines("HAVE_LAPACK_CONFIG_H")
    add_packages("openblas", { public = true })

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})

