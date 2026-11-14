# SIMD 层级架构（开发者参考）

面向用户的概述见 [README_zh.md — 功能特性](../README_zh.md#功能特性) 中的 SIMD 行、[TOOLS_zh.md](TOOLS_zh.md) 中的 `--simd-level` 选项，以及 [ENVIRONMENT_zh.md](ENVIRONMENT_zh.md#cosyvoice_simd_level) 中的 `COSYVOICE_SIMD_LEVEL` 运行时开关。本文覆盖内部结构：文件布局、能力模型、分派规则、AVX10 策略与构建期开关。

## 文件布局

| 文件 | 职责 |
|---|---|
| `src/simd-dispatch.h` | `simd_caps` 结构体、preset、`simd_dispatch<Kernel>`——不含任何 intrinsic 类型，调用方 TU 可安全包含 |
| `src/simd_detect.cpp` | CPUID/XCR0 特性探测 → 不可变 `g_simd_hw_caps` + 可写 `g_simd_caps`（初始与硬件能力一致）+ `COSYVOICE_SIMD_LEVEL` 环境变量初始化——全部定义在同一 TU，初始化顺序由定义顺序保证（仅 x86-64） |
| `src/simd-kernels.h` | kernel 类声明 + `simd_vec<C, N>` 向量 typedef + extern template 声明 |
| `src/simd-kernels-impl.h` | 全部 kernel 本体与支持助手，按 `simd_caps C` 模板化——**只**被层级 TU 包含 |
| `src/simd-math.h` | 向量 log/sincos 助手（128/256/512）——**只**被层级 TU 包含 |
| `src/simd/kernels_<tier>.cpp` | 每层一个 TU：定义 `SIMD_TIER_IMPL`，包含上述头文件，实例化本层的 preset |

每个层级 TU **只带自己那层的指令集编译标志**（GCC/Clang：`-msse4.2`、
`-msse4.2 -mavx`、……、`-mavx512f -mavx512bw -mavx512dq`、`-mavx10.1-256`；
MSVC：`/arch:SSE4.2`、`/arch:AVX`、`/arch:AVX2`、`/arch:AVX512`、
`/arch:AVX10.1 /vlen=256`）。这保证该对象里发出的每一条指令——无论是内联函数
还是编译器自动向量化——都被限制在本层指令集内，而所有调用方 TU 停留在
x86-64 基线。标量层在所有工具链上都不带任何标志。

kernel 调用点在链接期解析：调用方对象引用
`Kernel<simd_xxx>::run`（在 `simd-kernels.h` 中 `extern template` 声明），
由各层级对象提供定义。

## 能力模型

```cpp
struct simd_caps {          // 每类一个位，位域打包
    bool sse42, avx, fma3, avx2, avx512, avx10_1_256, avx10_1_512;
};
```

preset 是用作模板参数的 `constexpr simd_caps` 值：
`simd_none`、`simd_sse42`、`simd_sse42_fma`、`simd_avx`、`simd_avx2`、
`simd_avx512`、`simd_avx10_1_256`。

**preset 精确枚举自己所属的类**（除类本身真实蕴含的 legacy 位外不做向上继承）：
AVX10 的 preset 只带 `fma3` + 自己的位。因此共享代码路径在两层都显式列出全部
消费者——宏门与配套的 `if constexpr` 条件一致，例如

```cpp
#if defined(COSYVOICE_HAS_AVX) || defined(COSYVOICE_HAS_AVX2) || defined(COSYVOICE_HAS_AVX10_1_256)
    if constexpr (C.avx || C.avx10_1_256)
```

`COSYVOICE_HAS_*` 宏是**承重结构**：每个 intrinsic 块都由预处理器门控，没有
相应声明的平台（无 SIMDe 的非 x86）根本不会解析到它们。宏门必须与
`if constexpr` 条件保持一致——低类的共享代码块同时就是复用它的各个高类的
主循环/向量尾巴。

类边界之间使用**独立的 `if constexpr`，绝不用 `else`** 串联：每层的 512
或掩码 256 分支处理完自己的尾巴就 `return`，且 preset 的位互斥，
不存在落穿。`else` 会悬空到下一个宏门的实际残留物上，在
"AVX10-256 开启、SSE4.2/AVX/AVX2 关闭"这类配置下直接编译失败。

## 探测（`simd_detect.cpp`）

- CPUID leaf 1 → `sse42`、`osxsave`、`avx`、`fma3`。
- `XGETBV(0)` 门槛：任何 AVX 类位都要求 bit1+bit2（XMM/YMM 状态）。
- CPUID leaf 7 子叶 0 → `avx2`；`avx512` 需要 **F+DQ+BW+VL** 同时置位且
  `XCR0 & 0xE6 == 0xE6`（opmask + ZMM 状态）。尽管 512 分支自身不发
  128/256 位 EVEX，VL 仍是必需的——编译器自己的溢出代码会用
  EVEX xmm16+/ymm16+。只有部分子集的型号（Knights Landing：仅 F）落到 AVX2。
- **AVX10** — CPUID leaf 0x24H（独立于 leaf 7 的 legacy AVX2/AVX-512 枚举位，
  因为纯 AVX10 型号不置位这些位；但仍在上方 leaf 1 的 OSXSAVE+AVX 与
  XGETBV 门槛之后——任何 AVX10 型号都满足该门槛）：
  `EAX[7:0]` = 版本（1 = AVX10.1，2 = AVX10.2），
  `EBX bit16` = 256 位，`EBX bit17` = 512 位。256 类还要求
  `XCR0 & 0x26 == 0x26`（opmask）；512 类要完整 `0xE6`。版本 ≥ 1 即点亮
  两类；10.2 与 10.1 点亮相同的位。
- 探测结果缓存于不可变的 `g_simd_hw_caps`（一次性静态初始化）；可变的
  `g_simd_caps`（分派实际读取的对象）以同一值起步。Debug 构建启动时打印探测结果。

## 运行时控制 API

- x86-64 构建在 `cosyvoice.h` 暴露 `cosyvoice_get_simd_info()` /
  `cosyvoice_get_simd_level()` / `cosyvoice_set_simd_level()`（特性检测宏
  `COSYVOICE_SIMD_CONTROL_SUPPORTED`）。层级上限作用于**分派层级**而非硬件：
  setter 一次性计算 `simd_caps_for_level(g_simd_hw_caps, level)`
  （位于 `simd-dispatch.h`）并把结果写入可变的 `g_simd_caps`；分派热路径
  随后只剩对该值的一次 relaxed 原子加载 + 既有的"最强优先"链——例如
  AVX-512 机器上设 `COSYVOICE_SIMD_LEVEL_AVX2` 会选中 AVX2 层，设
  `COSYVOICE_SIMD_LEVEL_SCALAR` 强制标量层。`AUTO`（默认）恢复为无上限的
  硬件真值。
- 上限按每层一类对应：SSE42、AVX、AVX2、AVX10.1-256、AVX-512。AVX-512 档
  同时放行 AVX10-512（同一层）；AVX10-256 档只移除两个 512 位枚举来源，因此
  AVX10 能力部件保留 256 层、其余部件保留 AVX2 及以下。请求高于硬件能力或
  构建未包含的层级无害（分派自动钳制到可用最优层）。
- 语义：进程级全局、原子。封顶计算只在设置时执行一次；分派热路径仅是对
  `g_simd_caps`（C++17 `inline` 原子变量，各调用方 TU 共享同一实例）的一次
  relaxed load，相对 O(n) 内核可忽略。`g_simd_level` 只是供
  `cosyvoice_get_simd_level()` / `cosyvoice_simd_info_t.level` 读回用的元数据，
  不参与分派。修改影响后续内核调用；已进入的内核保持其入口时的层级。
- 环境变量：`COSYVOICE_SIMD_LEVEL` 在**库加载时**（`simd_detect.cpp` 的静态
  初始化，与探测结果同 TU，初始化顺序由定义顺序保证）读取一次，对所有
  消费者生效。取值（大小写不敏感）：`auto`、
  `scalar`（别名 `none`）、`sse42`（别名 `sse4.2`）、`avx`、`avx2`、
  `avx10-256`（别名 `avx10_256`、`avx10.1-256`、`avx10_1_256`）、`avx512`
  （别名 `avx-512`）。非法值静默忽略；进程启动后修改环境变量无效。
  优先级：显式 `cosyvoice_set_simd_level()`（如 CLI/Server 的 `--simd-level`）
  > 环境变量 > `AUTO`。
- 可用性：x86-64 始终导出这些符号，含 `COSYVOICE_NO_SIMD` 构建（此时
  `supported`/`current` 报 0、`scalar_only=true`，仅接受 `AUTO`/`SCALAR`）。
  非 x86（ARM64/SIMDe）构建什么都不导出——层级编译期固定——且头文件特性宏
  未定义；调用需用该宏保护。
- 层级枚举值是公开 ABI：`simd-dispatch.h` 中的内部 `simd_level` 必须与
  `cosyvoice_simd_level_t` 完全一致（`cosyvoice-simd.cpp` 里的
  `static_assert` 逐值校验）。

## 分派

`simd_dispatch<Kernel>(args...)` 是所有调用方 TU 共享的一个内联模板；x86 上它从可变的 `g_simd_caps`（硬件真值经运行时层级封顶，见上节）选择层级。顺序：
**512（legacy ∨ AVX10）→ AVX10.1-256 → AVX2 → AVX → SSE4.2 → 标量**。
构建期关闭的类被整个编出分派链（每个 case 外有
`#if defined(COSYVOICE_HAS_*)`），所以 CPU 总是落到可用层级里的最优档；
一个都不剩时抛异常。`COSYVOICE_NO_SIMD` 下分派直接坍缩为
`Kernel<simd_none>`。非 x86 上完全不走探测——直接返回
`Kernel<simd_sse42_fma>`（SIMDe 模拟），且这类目标上连标量层对象都不构建。

## AVX10 策略

AVX10.1 **512 位**部件由 **AVX-512 层的 kernel** 服务；不存在独立的 512
层级对象。依据（已验证）：

- MSVC：`/arch:AVX512` 与 `/arch:AVX10.1 /vlen=512` 产出**逐字节相同**的层级
  对象（反汇编逐行 diff 为空）。
- Clang：`-mavx512f -mavx512bw -mavx512dq` 与 `-mavx10.1-512` 的输出只差
  少量 VEX/EVEX 编码宽度*选择*；legacy 层的指令空间（VEX.128/256 +
  EVEX.512，F/BW/DQ/VL 子集，含 gather）是 AVX10.1-512 的严格子集。AVX10
  剔除的少数指令（MASKMOVDQU、VPOPCNTDQ 等）我们并未使用。
- 两条枚举路径的 XCR0 要求完全相同。

附带收益：完全不懂 AVX10 编译参数的老工具链（如 gcc < 14）也能服务
AVX10-512 的 CPU——构建该层只需要 `-mavx512*`。

AVX10.1 **256 位**类需要独立层级：k 掩码 256 位向量尾巴
（`_mm256_maskz_loadu_ps` / `_mm256_mask_storeu_ps`、`__mmask8`）与 EVEX
k 掩码版 log/sincos 助手（`_mm256_cmp_ps_mask`、`_mm256_cmp_epi32_mask`、
`_mm256_maskz_mov_ps`、`_mm256_mask_blend_ps`、`_mm256_mask_xor_epi32`）
无法由 AVX2 构建产出；而 256-only 的部件也绝不能跑 512 层（没有 ZMM 状态）。
该类复用共享的 AVX 8-wide 主体，仅替换尾巴。共享的载体是 typedef
`simd_vec<simd_avx10_1_256, 8>::mask = __mmask8`。

**AVX10.2** 不设层级：它是 10.1 的严格超集，且没有 kernel 使用 10.2 专属
指令（FP16 算术、`VFPCLASS`、bf16 助手、`V4FMADD` 等）。将来一旦用到，
恢复独立类的做法是：新 caps 位 + preset + 层级 TU + 在 10.1 case 之前插入
分派 case，并继续遵守"preset = 精确自身类"的规则。

## 构建期开关

| CMake 缓存项 | 编译定义 | 门控对象 |
|---|---|---|
| `COSYVOICE_NO_SIMD` | `COSYVOICE_NO_SIMD` | 关闭全部 SIMD（非 x86 且无 SIMDe 时自动开启） |
| `COSYVOICE_HAS_SCALAR` | `COSYVOICE_HAS_SCALAR` | 标量**分派**回退分支；x86 上标量层对象本身无条件构建，与本选项无关——非 x86 + SIMDe 时跳过（永远不会被分派到） |
| `COSYVOICE_HAS_SSE42` / `_AVX` / `_AVX2` / `_AVX512` | 同名 | 各 legacy 层级 |
| `COSYVOICE_HAS_AVX10_1` | `COSYVOICE_HAS_AVX10_1_256` | 仅 AVX10-256 层（外加编译器标志探针） |

- 关闭 SSE4.2 会级联关闭 AVX/AVX2/AVX-512（共享代码路径）；AVX10-256 类
  不在级联内（其实例化的分支不消费任何 SSE4.2 专属块）。
- 层级对象与所有库目标收到同一份**统一宏集**；层间差异完全由各自的编译
  标志特化。
- AVX10-256 探针（`check_cxx_source_compiles`，
  `-mavx10.1-256` / `/arch:AVX10.1 /vlen=256`）在不认识这些标志的工具链上
  静默去掉该类。
- LTO（`CMAKE_INTERPROCEDURAL_OPTIMIZATION`，默认 ON）还会在链接期折叠
  逐字节相同的层级实例化（MSVC ICF / GCC-Clang ICF）。混用 `/arch` 输入
  已验证安全：LTCG 按编译单元记录目标特性，逐函数按各自 ISA 生成代码。

## 非 x86

SIMDe（找到时）只模拟 SSE4.2+FMA3 这一类；检测与选项见 README 构建章的
SIMDe 小节。上述 128 位以上的所有内容在这些目标上都被编译排除
（AVX/AVX2/AVX-512/AVX10 的 `COSYVOICE_HAS_*` 宏永不定义），被宏门控的
intrinsic 体因此根本不会进入解析器。
