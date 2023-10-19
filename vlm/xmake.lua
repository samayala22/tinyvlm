add_requires("tbb", "eigen", "openmp")

target("libvlm")
    set_kind("static")
    add_packages("tbb", "eigen", "openmp")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})

target("vlm")
    set_kind("binary")
    set_default(true)
    add_rpathdirs("$ORIGIN") -- tbb dll must be in same dir as exe
    add_packages("openmp")
    add_deps("libvlm")
    set_runargs({"-i"}, {"../../../../config/elliptic.vlm"}, {"-m"}, {"../../../../mesh/elliptic_64x64.x"}, {"-o"}, {"../../../../results/elliptic.vtu"})
    add_files("dev/main.cpp")

target("vlm-tests")
    set_kind("binary")
    set_default(false)
    add_packages("openmp")
    add_deps("libvlm")
    add_files("dev/tests.cpp")