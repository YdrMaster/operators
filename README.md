# 算子库

跨平台高性能通用算子库。形式为 C 接口动态库。

采用二段式算子设计，每个算子都实现并对外暴露以下的 C 接口:

- 第一阶段：构造算子 Descriptor。用户提供的算子名称、硬件、以及算子配置（如计算的数据类型、计算排布等），相应模组会被 load 到硬件上。

  ```C
  void* createOpDescriptor(Device, void *config);
  ```

- 第二阶段：计算。根据一阶段的 Descriptor，执行相应计算，用户需要提供输入输出张量，以及硬件计算流（CPU 为 NULL）。

  ```C
  void op(void *descriptor, MutableTensor output, ConstTensor input, void *stream);
  ```

- 销毁 Descriptor。

  ```C
  void destroyOpDescriptor(void *descriptor);
  ```

## 一、使用说明

### 配置

#### 查看当前配置

```xmake
xmake f -v
```

#### 配置 CPU （默认配置）

```xmake
xmake f --cpu=true -cv
```

#### 配置 GPU

需要指定 CUDA 路径， 一般为 `CUDA_HOME` 或者 `CUDA_ROOT`。

```xmake
xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv
```

### 编译

```xmake
xmake
```

### 运行算子测试

```bash
cd operatorspy/tests
python operator_name.py
```

## 二、开发说明

### 目录结构

```bash
├── xmake.lua  # xmake 构建脚本
├── src
│   ├── devices
│   │   ├── [device_name]
│   │       ├── *.cc/.h # 特定硬件（如cpu、英伟达）通用代码
│   ├── ops
│   │   ├── utils.h  # 全算子通用代码 (如assert)
│   │   ├── [operator_name]  # 算子实现目录
│   │       ├── operator_name.cc/.h # 算子C接口
│   │       ├── [device_name]
│   │       │   ├── *.cc/.h/... # 特定硬件的算子实现代码
│   ├── *.h  # 核心结构体定义
│  
├── operatorspy  # Python封装以及测试脚本
    ├── tests
    │   ├── operator_name.py  # 测试脚本
    ├── *.py     # Python封装代码
```

### 增加新的硬件

- 在 `src/device.h` 和 `operatorspy/devices.py` 中增加新的硬件类型，注意两者需要一一对应；
- 在 `xmake.lua` 中增加新硬件的编译选项以及编译方式；
- 在 `src/ops/devices/[device_name]` 下编写特定硬件的通用代码；
- 实现该硬件的算子；

### 增加新的算子

- 在 `src/ops/[operator_name]` 增加创建/销毁算子描述符、算子计算的C接口；
- 在 `src/ops/[operator_name]/[device_name]` 增加算子在各硬件的实现代码；
- 在 `operatorspy/tests/[operator_name].py` 增加算子测试；
