# 算子库

## 配置

### 查看当前配置

```xmake
xmake f -v
```

### 配置 CPU

```xmake
xmake f --cpu=true -c -v
```

### 配置 GPU

```xmake
xmake f --nv-gpu=true --cuda=$CUDA_ROOT -c -v
```

**根据实际硬件修改 [arch](/xmake.lua#L32)。**

```lua
add_cuflags("-arch=sm_xx")
```

## 编译

```xmake
xmake
```
