# vLLM-SM75

[简体中文](README.md) | [English](README.en.md)

项目目标是持续跟进官方
[vLLM](https://github.com/vllm-project/vllm)，完善其对 SM75 的兼容支持，
并优化相关内核。

当前 vLLM-SM75 v0.1.0 基于官方 vLLM `v0.28.0`，已在 4 x Tesla T10
16 GiB、CUDA 12.9、TP4 环境中使用 Qwen3.8 27B FP8 完成验证。

## 功能与改进

- 新增 `flashqla_sm75` GDN prefill backend。
- 集成 image-owned FlashQLA-SM75 CUDA 扩展，只编译 `sm_75`，默认禁止
  runtime JIT，不依赖宿主机 `.so` 挂载。
- FlashInfer 从官方基线的 0.6.16.post3 升级到 0.6.18，并恢复 SM75
  full-attention 能力选择。
- GDN prefill 与 decode 独立控制；最终配置使用 FlashQLA-SM75 prefill 和
  Triton decode。
- 不支持的 GPU、dtype 或 GDN head dimension 自动回退到 Triton/FLA。
- 保留 Marlin FP8、FP8 E4M3 KV cache、prefix cache、async scheduling、
  `FULL_AND_PIECEWISE` CUDA Graph 和 CPU KV offload。
- FlashInfer sampler 在 SM75 上关闭，attention 仍使用 FlashInfer，sampling
  回退到 vLLM 原生实现。

FlashQLA 源码来自
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)，固定提交
`187b932dbd11940f0bcf52fb3675dd47fd69f313`。来源和许可证保留在
`vllm/third_party/flash_qla_sm75/`。

## 验证环境

| 组件 | 版本或配置 |
| --- | --- |
| vLLM-SM75 | v0.1.0 |
| 上游 vLLM | `v0.28.0` / `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| GPU | 4 x Tesla T10 16 GiB / SM75 |
| CUDA | 12.9 |
| PyTorch | 2.13.0 |
| FlashInfer | 0.6.18，无 `flashinfer-jit-cache` |
| 验证模型 | Qwen3.8 27B FP8 |
| Tensor parallel | TP4 / PYNCCL |
| GDN | FlashQLA-SM75 prefill + Triton decode |

## 实测结果

测试保持模型、TP、attention、KV、sampling 和服务资源一致，仅比较 GDN
prefill 路线。

| 指标 | vLLM v0.28.0 生产基线 | vLLM-SM75 v0.1.0 | 变化 |
| --- | ---: | ---: | ---: |
| Cold TTFT | 1.9562 s | 1.7082 s | **提升 12.68%** |
| Prefix-cached TTFT | 0.5064 s | 0.4378 s | **提升 13.55%** |
| Decode | 39.6289 token/s | 39.6306 token/s | 基本持平 |
| Cached tokens | 1,568 | 1,568 | 不变 |

确定性输出、streaming、tool calling、reasoning parser、多模态、prefix cache、
FP8 KV 和 8 GiB CPU KV offload 均通过验证。GDN 配置的 GPU KV 容量为 506,209
tokens，并观测到 821,297,152 bytes GPU 到 CPU KV 迁移。

## 快速复现

### 1. 克隆仓库

```bash
git clone https://github.com/fishensw/VLLM-SM75.git
cd VLLM-SM75
```

### 2. 构建镜像

准备兼容 vLLM 0.28.0、CUDA 12.9、PyTorch 2.13.0 和 SM75 的基础镜像：

```bash
export BASE_IMAGE='your-registry.example/vllm-openai:v0.28.0-cu129-sm75'

docker build \
  --file docker/Dockerfile.vllm-sm75-v0.1.0 \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg BASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  --tag vllm-sm75-v0.1.0 \
  .
```

Dockerfile 会编译 SM75 扩展、检查 `cuobjdump` 架构、验证 `runtime` 版本，并在
构建阶段运行必要的 backend selector tests。

### 3. 启动服务

```bash
export VLLM_API_KEY='replace-with-your-api-key'
docker volume create vllm-hf-cache

docker run --detach --rm \
  --name vllm-sm75 \
  --gpus all \
  --ipc=host \
  --publish 8000:8000 \
  --volume vllm-hf-cache:/root/.cache/huggingface \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env FLASH_QLA_SM75_ALLOW_JIT=0 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_USE_NCCL_SYMM_MEM=0 \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  --entrypoint vllm \
  vllm-sm75-v0.1.0 \
  serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name VLLM-Qwen3.8-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --gpu-memory-utilization 0.87 \
  --tensor-parallel-size 4 \
  --max-model-len auto \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --gdn-prefill-backend flashqla_sm75 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'

docker logs --follow vllm-sm75
```

日志显示服务就绪后按 `Ctrl+C` 退出日志跟踪，再验证 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  --header "Authorization: Bearer $VLLM_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{"model":"VLLM-Qwen3.8-27B","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

以上参数对应本项目的 4 x Tesla T10、TP4、Qwen3.8 27B FP8 验证配置。更换
GPU 数量、模型或可用显存后，再相应调整 TP、模型长度、并发和显存利用率。

## License

vLLM 修改继续遵循上游 Apache-2.0 License。FlashQLA-SM75 文件保留其原始
MIT License 和来源说明。
