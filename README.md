# 算子库

跨平台高性能通用算子库。形式为C接口动态库。

采用三段式算子设计:

- 第一阶段：构造Operator。用户提供的算子名称、硬件、以及算子配置（如计算的数据类型、计算排布等）

```C
Op op_create(Device, Optype, void *config);
```

- 第二阶段：构造Kernel。根据一阶段的Operator，用户提供的运行时相关参数（如cuda stream等）

```C
Kn kn_load(Op, void *rt_ctx);
```

- 第三阶段：构造函数。根据二阶段的Kernel，返回算子计算的C函数指针

```C
void *fn_get(Kn);
```

## 一、使用说明

### 配置

#### 查看当前配置

```xmake
xmake f -v
```

#### 配置 CPU （默认配置）

```xmake
xmake f --cpu=true -c -v
```

#### 配置 GPU

需要指定 CUDA 路径， 一般为 `CUDA_HOME` 或者 `CUDA_ROOT`

```xmake
xmake f --nv-gpu=true --cuda=$CUDA_HOME -c -v
```

**根据实际硬件修改 [arch](/xmake.lua#L32)。**

```lua
add_cuflags("-arch=sm_xx")
```

### 编译

```xmake
xmake
```

### 编辑环境变量

将编译好 `liboperators.so` 所在的目录路径添加到 `LD_LIBRARY_PATH` 环境变量中。

```bash
export LD_LIBRARY_PATH=PATH_TO_SHARED_LIBRARY:$LD_LIBRARY_PATH
```

### 运行算子测试

```bash
cd operatorspy/tests
python operator_name.py 
```

## 二、开发说明

### 目录结构

```bash
operatorspy
├── xmake.lua  # xmake 构建脚本
├── src
│   ├── ops
│   │   ├── c_interface  # 算子c接口
│   │   │   ├── [device_name]
│   │   │   |   ├── *.cc/.h/..  # 特定硬件的c接口代码
│   │   ├── [operator_name]  # 算子实现目录
│   │       ├── [device_name]
│   │       |   ├── *.cc/.h/... # 特定硬件的实现代码
│   │       ├── *.cc/.h/...  # 通用代码
│   ├──  utils.cc/.h  # common工具代码
│   ├──  *.cc/.h  # operators库接口
|
├── operatorspy  # Python封装以及测试脚本
    ├── tests    
    │   ├── operator_name.py  # 测试脚本
    ├── *.py     # Python封装代码
```

### 增加新的硬件

- 在 `src/device.h` 和`operatorspy/devices.py`中增加新的硬件类型。注意两者需要一一对应。
- 在 `xmake.lua` 中增加新硬件的编译选项以及编译方式。
- 在 `src/ops/c_interface/[device_name]` 下编写特定硬件的c接口代码。
- 在 `src/operators.c` 中增加新硬件的c接口调用。

### 增加新的算子

- 在 `src/optype.h` 和`operatorspy/operators.py`中增加新的硬件类型。注意两者需要一一对应。
- 在 `src/optype.h` 中定义算子计算函数的签名。
- 在 `src/ops/c_interface/[device_name]` 的c接口代码中增加该算子Operator和Kernel的构建方式。
- 在 `src/ops/[operator_name]/[device_name]` 增加算子在各硬件的实现代码。
- 在 `operatorspy/tests/` 增加算子测试。
