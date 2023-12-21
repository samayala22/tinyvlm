add_requires("tbb", "eigen", "openmp")

target("backend-avx2")
    set_kind("static")
    set_default(false)
    add_vectorexts("avx2", "fma")
    add_packages("tbb", "eigen", "openmp")

    add_includedirs("../../include")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})

