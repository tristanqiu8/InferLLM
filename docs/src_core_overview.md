# InferLLM `src/core` 代码总览

## 核心职责
- `src/core/graph.cpp`：管理 token 执行流程。`Graph::execute` 会按批次设置输入、复用工作区缓冲区并同步设备输出，再把 logits 返回给上层。
- `src/core/model_imp.cpp`：负责把配置、词表和计算图串起来，并驱动 prefill / decode，使推理流程贯通。
- `src/core/op.h`：`OprModuleBase` 定义模块化包装，把多个算子串成一段子图，统一推断输出形状与临时内存需求。

```cpp
// src/core/graph.cpp:166
void Graph::execute(
        std::vector<int32_t> in_token, std::vector<float>& logist,
        uint32_t nr_past, bool prefill) {
    if (m_input->dims() == 0 || !same_input_shape(in_token)) {
        m_input->set_shape({in_token.size()}, DType::Int32);
        size_t len = get_workspace_in_byte();
        if (m_workspace->ptr() == nullptr) {
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        } else if (m_workspace->ptr() && len > m_workspace->length()) {
            m_device->free_device(m_workspace->ptr());
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        }
    }
    ...
    m_device->sync();
    m_output->recall_data();
}
```

## 设备与并发层
- `src/core/device.h` / `src/core/device.cpp`：抽象统一/分离内存模型并暴露 `Kernel` 句柄。`CPUDevice` 维护按大小分类的内存池；`GPUDevice` 封装 CUDA 资源并提供异步拷贝接口。
- `src/core/thread_pool.cpp`：实现 CPU 内核使用的线程池，可按需激活/休眠，避免空转。

```cpp
// src/core/device.cpp:34
void* CPUDevice::allocate(size_t len) {
    auto it = m_free_memory.lower_bound(len);
    void* ptr = nullptr;
    if (it != m_free_memory.end() && it->second.size() > 0) {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.size() < 1) {
            m_free_memory.erase(it);
        }
    } else {
        ptr = aligned_alloc(len);
        m_alloc_memory[ptr] = len;
    }
    return ptr;
}
```

## Tensor 与 KV 缓存
- `src/core/tensor.h` / `src/core/tensor.cpp`：`Tensor` 记录形状、步长、dtype 与所有者引用计数，按需 mmap 或分配显存/主内存，权重在加载时可以根据算子需求做预处理（例如 int4 重排）。
- `src/core/kvstorage.h` / `src/core/kvstotage.cpp`：`KvStorage` 继承 `Tensor` 用于注意力缓存，追踪当前写入位置，不够时自动扩容并保持连续的 key/value 区间。

```cpp
// src/core/kvstotage.cpp:53
TensorState KvStorage::prepare_data_with_length(uint32_t len) {
    Tensor::prepare_data();
    if (m_store_id + len >= m_curr_id) {
        auto shape = this->shape();
        shape[0] = m_curr_id + KvStorageConfig::KV_STEP;
        size_t old_len = length_in_byte();
        void* old_ptr = ptr();
        set_shape(shape, dtype());
        size_t len = length_in_byte();
        auto data = device()->aligned_alloc(len);
        device()->device2device_copy(data, old_ptr, old_len);
        device()->aligned_free(old_ptr);
        set_shared_memory(data, len);
        m_curr_id += KvStorageConfig::KV_STEP;
    }
    ...
    return TensorState::Own;
}
```

## 计算图模块
- 注意力模块（`graph.h` 中模板定义，`op.cpp` 中实现）组合 QKV matmul、RoPE、KV 缓存更新与输出投影，派生类覆盖 LLaMA、GLM 及 GLM2 多 Query 版本。
- 前馈模块（`LlamaFFNModule`、`GlmFFNModule`、`Glm2FFNModule`）还原各模型的激活链路（SiLU gating、GELU、双激活）并封装成算子序列。
- `HeadModule` 在 prefill 阶段跳过最终 logits，只在生成阶段执行 layer norm + 最后一 token 的 matmul。
- `Graph::collect_weights` 与 `Graph::get_weight_alias` 汇总模块权重并重写层号别名，保证加载阶段能对齐转换后的权重命名。

## 基础算子
- `OpBase` 统一 `pre_execute` / `execute` / `end_execute` 生命周期，并管理输入输出的引用计数。派生算子直接调用后端 `Kernel`。
- `MatMul` / `MatMulLast` 支持 float/int8/int4 权重，可对 int4 权重执行 `MatmulInt4WeightReorder` 打包以匹配优化核。
- 注意力算子负责生成 QKV，申请临时空间（保存 Q、K、V 以及 qk 中间结果），应用 RoPE、softmax，并与 KV 缓存矩阵相乘生成输出。
- 其他算子（`LayerNorm`、`Embedding`、`Elemwise`、`SpliteHalfActiveMul`、`SoftMax`、`DiagMask`）是对特定 kernel 的轻量封装。

```cpp
// src/core/op.cpp:243
std::vector<size_t> MatMul::preprocess_weight(Tensor* tensor, void* src, void* dst) {
    INFER_ASSERT(tensor->dtype() == DType::Int4, "only support optimized int4 kernel");
    auto weight_shape = tensor->shape();
    size_t M = weight_shape[0];
    size_t N = weight_shape[1];
    auto kernel = get_kernel();
    kernel->operator()<KernelID::MatmulInt4WeightReorder>(M, N, dst, src, PACK_SIZE);
    size_t block_m = M / PACK_SIZE;
    m_weight_packed = true;
    return {block_m, N * PACK_SIZE};
}
```

## 推理流程粘合层
- `Model` 对外暴露 `load`、`init`、`prefill`、`decode`、`decode_iter` 等接口，本身只是 `ModelImp` 的薄封装。
- `ModelImp` 选择设备后初始化 `Graph`，并维护运行态：上下文长度、KV 缓存重置、抽样队列与计时统计。
- `ModelImp::tokenize` 通过动态规划最大化子串得分，按需插入 BOS；`decode_summary` 汇总延迟与吞吐指标。

## 调用关系图

```mermaid
flowchart TD
    A[Model::decode] --> B[ModelImp::decode]
    B --> C[tokenize]
    B --> D[Graph::post_tokenize]
    B --> E[Graph::execute]
    E --> F[OprModuleBase::execute]
    F --> G[OpBase::execute]
    B --> H[sample_and_update]
    H --> I[llama_sample_top_p_top_k]
```

```mermaid
flowchart TD
    M[Graph::execute] --> N[get_workspace_in_byte]
    M --> O[m_input->prepare_data]
    M --> P[Device::host2device_copy]
    M --> Q[Module Loop]
    Q --> R[Module::execute]
    R --> S[OpBase::execute]
    M --> T[Device::device2host_copy]
    M --> U[Device::sync]
```
