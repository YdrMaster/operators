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

target("common")
    set_kind("static")
    set_languages("c11")
    add_files("src/utils.c")
    if is_mode("debug") then
        add_cxflags("-g")
    end
target_end()

if has_config("cpu") then
add_defines("ENABLE_CPU")
target("cpu")
    set_kind("static")
    set_languages("cxx17")
    add_files("src/cpu/*.cc")
    add_deps("common")
    if is_mode("debug") then
        add_cxflags("-g")
    end
target_end()

end

if has_config("nv-gpu") then

add_defines("ENABLE_NV_GPU")
target("nv-gpu")
    set_kind("static")
    set_languages("cxx17")
    add_cuflags("-arch=sm_80", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
    add_files("src/nv_gpu/*.cu")
    set_toolchains("cuda")
    set_policy("build.cuda.devlink", true)
    add_deps("common")
    if is_mode("debug") then
        add_cxflags("-g")
    end
target_end()

end

target("operators")
    set_kind("static")
    set_languages("c11")
    add_files("src/operators.c")
    add_deps("common")
if has_config("cpu") then
    add_deps("cpu")
end
if has_config("nv-gpu") then
    add_deps("nv-gpu")
end
if is_mode("debug") then
    add_cxflags("-g")
end
target_end()

target("main")
    set_kind("binary")
    set_languages("cxx17")
    add_files("src/main.cpp")
    add_deps("operators")
    if is_mode("debug") then
        add_cxflags("-g")
    end
target_end()
