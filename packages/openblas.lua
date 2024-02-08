package("openblas_custom")
    set_homepage("http://www.openblas.net/")
    set_description("OpenBLAS is an optimized BLAS library based on GotoBLAS2 1.13 BSD version.")
    set_license("BSD-3-Clause")

    if is_plat("windows") then
        if is_arch("x64", "x86_64") then
            add_urls("https://github.com/OpenMathLib/OpenBLAS/releases/download/v$(version)/OpenBLAS-$(version)-x64.zip")
            add_versions("0.3.26", "859c510a962a30ef1b01aa93cde26fdb5fb1050f94ad5ab2802eba3731935e06")
        elseif is_arch("x86") then
            add_urls("https://github.com/OpenMathLib/OpenBLAS/releases/download/v$(version)/OpenBLAS-$(version)-x86.zip")
            add_versions("0.3.26", "9c3d48c3c21cd2341d642a63ee8a655205587befdab46462df7e0104d6771f67")
        end
    else
        add_urls("https://github.com/OpenMathLib/OpenBLAS/releases/download/v$(version)/OpenBLAS-$(version).tar.gz")
        add_versions("0.3.26", "4e6e4f5cb14c209262e33e6816d70221a2fe49eb69eaf0a06f065598ac602c68")
    end

    add_configs("shared", {description = "Build shared library.", default = true, type = "boolean", readonly = is_plat("windows")})
    add_configs("lapack", {description = "Build LAPACK", default = true, type = "boolean", readonly = is_plat("windows")})
    add_configs("dynamic_arch", {description = "Enable dynamic arch dispatch", default = (is_plat("linux") or is_plat("windows")), type = "boolean", readonly = not is_plat("linux")})
    add_configs("openmp", {description = "Compile with OpenMP enabled", default = (is_plat("windows") or is_plat("linux")), type = "boolean", readonly = not is_plat("linux")})
    
    if not is_plat("windows") then
        add_deps("cmake")
    end
    if is_plat("linux") then
        add_extsources("apt::libopenblas-dev", "pacman::libopenblas")
        add_syslinks("pthread")
    end
  
    on_load("linux", function (package)
        if package:config("openmp") then package:add("deps", "openmp") end
    end)

    on_load("windows|x64", "windows|x86", function (package)
        package:add("defines", "HAVE_LAPACK_CONFIG_H")
    end)

    on_install("windows|x64", "windows|x86", function (package)
        os.cp("bin", package:installdir())
        os.cp("include", package:installdir())
        os.cp(path.join("lib", "libopenblas.lib"), path.join(package:installdir("lib"), "openblas.lib"))
        package:addenv("PATH", "bin")
    end)

    on_install("macosx", "linux", function (package)
        local configs = {"-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",  "-DNOFORTRAN=ON"}
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DDYNAMIC_ARCH=" .. (package:config("dynamic_arch") and "ON" or "OFF"))
        table.insert(configs, "-DUSE_OPENMP=" .. (package:config("openmp") and "ON" or "OFF"))
        if package:is_plat("macosx") and package:is_arch("arm64") then
            table.insert(configs, "-DTARGET=VORTEX") 
            table.insert(configs, "-DBINARY=64")
        end
        if package:config("lapack") then
            table.insert(configs, "-DC_LAPACK=ON")
            table.insert(configs, "-DBUILD_LAPACK_DEPRECATED=OFF")
        else
            table.insert(configs, "-DONLY_CBLAS=ON")
            table.insert(configs, "-DBUILD_WITHOUT_LAPACK=ON")
        end
        import("package.tools.cmake").install(package, configs)

        os.mv(package:installdir() .. "/include/openblas/*" , package:installdir("include"))
        os.rm(package:installdir("include/openblas"))
    end)

    on_test(function (package)
        assert(package:check_csnippets({test = [[
            void test() {
                double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
                double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
                double C[9] = {0.8,0.2,0.5,-0.3,0.5,0.2,0.1,0.4,0.1};

                cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,3,3,2,1,A,3,B,3,2,C,3);
            }
        ]]}, {includes = {"cblas.h"}}))

        if package:config("lapack") then
            assert(package:check_csnippets({test = [[
                void test() {
                    double A[9] = {0.8,0.2,0.5,-0.3,0.5,0.2,0.1,0.4,0.1};
                    double x[3] = {1.0,2.0,3.0};
                    lapack_int ipiv[3];

                    LAPACKE_dgesv(LAPACK_COL_MAJOR, 3, 1, A, 3, ipiv, x, 3);
                }
            ]]}, {includes = {"lapacke.h"}}))
        end
    end)
