add_requires("tbb", "openmp", "eigen")

target("backend-avx2")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("tbb", "openmp", "eigen")

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})

