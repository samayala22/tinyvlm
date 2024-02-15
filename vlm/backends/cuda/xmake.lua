add_requires("cuda", {configs={utils={"cublas", "cusolver"}}})

target("backend-cuda")
    set_kind("static")
    set_default(false)
    add_packages("cuda")
    add_deps("backend-cpu")

    add_includedirs("../../include")
    add_files("src/*.cu")
    add_includedirs("include", {public = true})
