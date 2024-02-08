add_requires("taskflow_custom")

target("libvlm")
    set_kind("static")
    add_packages("taskflow_custom", {public = true})
    
    for _,name in ipairs(backends) do
        if has_config(backend_option(name)) then
            add_defines(backend_defines(name))
            add_deps(backend_deps(name))
        end
    end
    
    add_files("src/*.cpp")
    add_includedirs("include", {public = true})
    
target("vlm")
    set_kind("binary")
    set_default(true)
    add_rpathdirs("$ORIGIN")
    add_deps("libvlm") -- core library
    set_runargs({"-i"}, {"../../../../config/elliptic.vlm"}, {"-m"}, {"../../../../mesh/elliptic_64x64.x"}, {"-o"}, {"../../../../results/elliptic.vtu"})
    add_files("dev/main.cpp")

-- xmake run vlm -i ../../../../config/elliptic.vlm -m ../../../../mesh/elliptic_128x128.x -o ../../../../results/rec.vtu
