add_requires("openmp")

target("libvlm")
    set_kind("static")
    -- add_options("avx2", "cuda")

    -- there has to be a better way to do this
    -- if get_config("avx2") == "y" then
    --     add_deps("backend-avx2")
    -- end

    -- if get_config("cuda") == "y" then
    --     add_deps("backend-cuda")
    -- end

    add_defines("VLM_AVX2")
    add_deps("backend-avx2")
    add_defines("VLM_CUDA")
    add_deps("backend-cuda")

    add_packages("openmp")
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})
    
target("vlm")
    set_kind("binary")
    set_default(true)
    add_rpathdirs("$ORIGIN") -- tbb dll must be in same dir as exe
    add_packages("openmp") -- need to add this or doesnt compile
    add_deps("libvlm") -- core library
    set_runargs({"-i"}, {"../../../../config/elliptic.vlm"}, {"-m"}, {"../../../../mesh/elliptic_90x90.x"}, {"-o"}, {"../../../../results/elliptic.vtu"})
    add_files("dev/main.cpp")

target("vlm-tests")
    set_kind("binary")
    set_default(false)
    add_packages("openmp")
    add_deps("libvlm")
    add_files("dev/tests.cpp")