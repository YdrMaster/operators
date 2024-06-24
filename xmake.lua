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

option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Ascend NPU kernel")
    add_defines("ENABLE_ASCEND_NPU")
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

if has_config("ascend-npu") then
    add_defines("ENABLE_ASCEND_NPU")
    local ascend_home = os.getenv("ASCEND_HOME")
    if ascend_home then
        -- Add include dirs
        add_includedirs(ascend_home .. "/include")
        add_includedirs(ascend_home .. "/include/aclnn")
        -- Add shared lib
        add_linkdirs(ascend_home .. "/lib64")
        add_links("libascendcl.so")
        add_links("libnnopbase.so")
        add_links("libopapi.so")    
        add_linkdirs(ascend_home .. "/../../driver/lib64/driver")
        add_links("libascend_hal.so")
    else 
        raise("ASCEND_HOME environment variable is not set!")
    end
    target("ascend-npu")
        set_kind("shared")
        -- Other configs
        set_languages("cxx17")
        add_cxflags("-lstdc++ -Wall -Werror")
        add_files("src/devices/ascend/*.cc")
        -- npu
        add_files("src/ops/*/ascend/*.cc")
        
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
if has_config("ascend-npu") then
    add_deps("ascend-npu")
end
target_end()

target("main")
    set_kind("binary")
    set_languages("c11")
    add_files("src/main.c")
    add_deps("operators")
target_end()
