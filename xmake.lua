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
    set_kind("shared")
    set_languages("cxx17")
    add_files("src/devices/cpu/*.cc")
    add_files("src/ops/*/cpu/*.cc")
target_end()

end

if has_config("nv-gpu") then

add_defines("ENABLE_NV_GPU")
target("nv-gpu")
    set_kind("shared")
    set_languages("cxx17")
    add_cuflags("-arch=sm_80", "--expt-relaxed-constexpr", "--allow-unsupported-compiler",{force = true})
    add_files("src/ops/*/cuda/*.cu")
    set_toolchains("cuda")
    set_policy("build.cuda.devlink", true)
target_end()

end

target("operators")
    set_kind("shared")
    set_languages("cxx17")
    add_files("src/ops/*/*.cc")
if has_config("cpu") then
    add_deps("cpu")
end
if has_config("nv-gpu") then
    add_deps("nv-gpu")
end
target_end()

target("main")
    set_kind("binary")
    set_languages("c11")
    add_files("src/main.c")
    add_deps("operators")
target_end()
