add_requires("eigen")

target("vlm")
    set_kind("binary")
    set_runargs({"-i"}, {"../../../../config/elliptic.vlm"}, {"-m"}, {"../../../../mesh/elliptic_python.xyz"}, {"-o"}, {"../../../../results/elliptic.vtu"})
    add_packages("eigen")
    -- add_packages("tbb", "lz4")
    add_files("src/*.cpp")
    add_files("dev/*.cpp")
    add_includedirs("include", {public = true})