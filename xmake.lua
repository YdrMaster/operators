add_rules("mode.debug", "mode.release")

option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable cpu kernel")
    add_defines("ENABLE_CPU")
option_end()

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Nvidia GPU kernel")
    add_defines("ENABLE_NV_GPU")
option_end()

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end

if has_config("cpu") then

    add_defines("ENABLE_CPU")
    target("cpu")
        set_kind("static")

        if not is_plat("windows") then
            add_cxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cpu/*.cc", "src/ops/*/cpu/*.cc")
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    target_end()

end

if has_config("nv-gpu") then

    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
        set_policy("build.cuda.devlink", true)

        set_toolchains("cuda")
        add_links("cublas")
        add_cugencodes("native")

        if is_plat("windows") then
            add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        else
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
        end

        set_languages("cxx17")
        add_files("src/ops/*/cuda/*.cu")
    target_end()

end

target("operators")
    set_kind("shared")

    if has_config("cpu") then
        add_deps("cpu")
    end
    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end

    set_languages("cxx17")
    add_files("src/ops/*/operator.cc")
target_end()

target("main")
    set_kind("binary")
    add_deps("operators")

    set_languages("c11")
    add_files("src/main.c")
target_end()
