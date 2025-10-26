# 接入新硬件（示例：NPU）指南

## 可直接复用的部分
- `Model` / `ModelImp`：推理状态管理、解码流程、采样逻辑已通过抽象接口完成，新增硬件无需改动。参见 `src/core/model_imp.h:39`、`src/core/model_imp.cpp:12`。
- `Graph` 与模块体系：构图、权重加载、别名映射、模块执行(`src/core/graph.cpp:166`、`:280`)已经与具体设备解耦，可原样复用。
- 算子模块与高层逻辑：注意力/前馈/LayerNorm 等模块及算子调用序列(`src/core/op.cpp:63` 起)不需要修改，只要新硬件内核实现与接口保持一致。
- 通用工具：`Tensor`、`KvStorage`、`ThreadPool` 等数据结构和内存管理器仍可沿用。

## 需要新增或扩展的部分
- **设备封装 (`Device`)**  
  - 新增 `NPUDevice` 继承 `Device`（参考 `GPUDevice`，路径 `src/core/device.h:71`、`src/core/device.cpp:88`）。  
  - 实现内存分配/释放、Host↔Device 拷贝、同步、对齐分配等接口，并在构造时初始化对应的 `Kernel` 对象。  
  - 根据硬件特性覆写 `unified_memory()`，让 `Tensor::read_data_from_file` 自动选择正确的数据流。

```cpp
// 示例：NPUDevice 骨架
class NPUDevice : public Device {
public:
    explicit NPUDevice(int device_id) {
        NPU_CHECK(npuSetDevice(device_id));
        m_kernel = make_unique<Kernel>(KernelType::NPU);
        m_kernel->set_handle(&m_handle);
    }

    void* allocate(size_t len) override {
        void* ptr = nullptr;
        NPU_CHECK(npuMalloc(&ptr, len));
        m_alloc_memory[ptr] = len;
        return ptr;
    }

    void free_device(void* ptr) override {
        INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "unknown ptr");
        NPU_CHECK(npuFree(ptr));
        m_alloc_memory.erase(ptr);
    }

    void host2device_copy(void* dst, const void* src, size_t size, bool async = false) override {
        NPU_CHECK(npuMemcpy(dst, src, size, async ? NPU_MEMCPY_H2D_ASYNC : NPU_MEMCPY_H2D));
    }
    ...
private:
    npuHandle_t m_handle{};
};
```

- **内核实现 (`Kernel`)**  
  - 在 `src/kern` 下新建目录（如 `npu/`），实现 `Kernel` 中被调用的所有算子入口（`MatmulInt4Float`、`SoftmaxFloat`、`RopeFloat`、`HeadBatchedMatmulFloat` 等）。  
  - 可参考 `kern/naive` 做纯软件 fallback，再针对 NPU SDK 做高性能版本。

```cpp
// 示例：在 Kernel 中接入 NPU matmul
template <>
void Kernel::operator()<KernelID::MatmulFloatFloat>(
        float* dst, const float* weight, const float* bias, const float* src,
        uint32_t M, uint32_t N, uint32_t K, void* workspace, uint32_t workspace_len) {
    npu_matmul(m_handle, dst, weight, bias, src, M, N, K, workspace, workspace_len);
}
```

- **构建配置**  
  - 修改 `CMakeLists.txt` 将 NPU 内核源文件编译进项目，必要时添加编译选项、链接硬件 SDK。

```cmake
# 示例：增加 NPU Kernel 目标
add_library(kern_npu
    src/kern/npu/kernel.cpp
    src/kern/npu/matmul.cpp
    src/kern/npu/softmax.cpp
)
target_link_libraries(kern_npu PRIVATE npu_runtime)
target_include_directories(kern_npu PRIVATE ${NPU_RUNTIME_INCLUDE_DIR})
```

- **设备选择逻辑**  
  - 在 `ModelImp` 构造函数中增补设备分支：`else if (device_type == "NPU")`，创建 `NPUDevice` 实例（`src/core/model_imp.h:44`）。

## 建议流程
1. 先基于 `kern/naive` 实现一套符合接口的 NPU 版本，用于功能验证。  
2. 完善 `NPUDevice` 与 `Kernel` 后，在配置文件或命令行中添加 `device_type=NPU`。  
3. 通过现有测试或示例应用验证推理结果，然后逐步优化数据搬运和算子性能。
