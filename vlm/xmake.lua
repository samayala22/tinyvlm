add_requires("openmp")

target("libvlm")
    set_kind("static")

    for _,name in ipairs(backends) do
        if has_config(backend_option(name)) then
            add_defines(backend_defines(name))
            add_deps(backend_deps(name))
        end
    end
    
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