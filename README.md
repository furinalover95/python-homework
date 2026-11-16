# Project 1：使用 NumPy 复现图像编码器

本项目使用给定图像和模型参数，通过 NumPy 实现 OpenCLIP ViT-B-32 图像编码器的前向计算。程序输出须与项目提供的参考结果数值匹配。本练习不涉及模型训练；完成的编码器可作为最终项目 `szu_python_openclip` 的图像塔。

## 上传包资源说明

本压缩包包含已完成代码、README、实验报告和 10 张测试图像。根据作业提交要求，未包含较大的模型权重 `weights/image_encoder.npz`，也未包含参考数据 `assets/reference_outputs.npz`。如需在本地重新运行完整检查或生成模型输出，请从课程原始附件取回这两个文件，并分别放回上述路径。没有模型权重时，`run.py` 无法执行；没有参考数据时，`check.py --stage all` 无法完成参考值比较。

## 快速开始

请在本目录打开终端。需要 Python 3.9 或更高版本。

```bash
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

处理自己的图像时，传入图像路径：

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`check.py --stage all` 会依次检查资源、预处理、图像块、嵌入 tokens、第 0 个 Transformer block、最终 tokens、输出向量和批次顺序。此项目的完整数值检查已通过。`run.py` 会按输入顺序保存图像 ID 和输出向量。

## GitHub

项目代码仓库：[furinalover95/python-homework](https://github.com/furinalover95/python-homework)。报告所用的深圳大学 LaTeX 模板仓库：[Experiment-Report-Template-SZU](https://github.com/ChaoFan996/Experiment-Report-Template-SZU)。

## 输入与输出

- `ImageEncoder` 输入：预处理后的 float32 数组 `[B,3,224,224]`。
- 经过 12 个 Transformer block 后的图像 tokens：`[B,50,768]`，包含 1 个 CLS token 和 49 个 patch tokens。
- 最终输出 token：每张图像对应一个归一化的 float32 向量，形状为 `[B,512]`。
- 本作业中“输出 token”专指最终的 512 维向量；`last_tokens` 指完整的 50 行内部序列。

## 项目文件

- `image_encoder.py`：六组 NumPy 前向计算实现。
- `helpers.py`：提供 LayerNorm、精确 GELU、注意力权重转换和单位向量缩放。
- `preprocess_image.py`：提供图像缩放、中心裁剪和像素归一化。
- `runtime.py`、`run.py`、`check.py`：资源加载、命令行运行和数值检查。
- `weights/image_encoder.npz`：模型的 152 个视觉参数张量；不含文本模型参数。
- `assets/images/`、`assets/images.json`：10 张本地图像及其 ID/来源信息。
- `assets/reference_outputs.npz`：独立 FP32 中间结果与输出参考，仅供检查使用。
- `config.json`、`weights/manifest.json`：模型结构参数及资源指纹。
- `transformer_steps.py`：用于逐步观察 Transformer 算法的微型示例。
- `lecture-demos/`：课堂 NumPy 示例和配套材料。
- `practice/project1_exercises.pdf`：12 道 NumPy 练习题，不含答案。
- `PPT-CODE-MAP.txt`：76 页课程讲义到项目代码的对应关系。
- `experiment_report.pdf`、`experiment_report.tex`：实验报告 PDF 与可编辑 XeLaTeX 源文件。

## 已完成的实现

所有待完成部分均位于 `image_encoder.py`，实现了以下六项任务。代码中补充了形状、轴变换、权重布局和残差路径的注释。

| 任务 | 方法 | 实现内容 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | 将 `[B,3,224,224]` 转为 `[B,49,3072]`，保持 patch 顺序 |
| TODO-02 | `MultiHeadAttention.split_heads` | 将 `[B,N,D]` 转为 `[B,H,N,D/H]` |
| TODO-03 | `MultiHeadAttention.__call__` | 计算注意力分数、加权 V、合并 head 并执行输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 计算两条残差、Norm 2、GELU 和两层 MLP |
| TODO-05 | `ImageEncoder.embed` | 投影 patch，拼接 CLS token 并加入位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 选取 CLS、执行 post-Norm、投影并归一化 |

## 课堂示例

运行微型 Transformer 示例：

```bash
python transformer_steps.py
```

该示例使用 3 行 token、4 个特征、1 个 head 和 6 个 MLP 隐藏值，以小型合成参数展示行均值、中心化、方差、Norm、Q/K/V、注意力分数与权重、加权值、投影、残差和 MLP。它用于说明计算过程，不是模型检查点，也不替代 `image_encoder.py` 中的完整实现。

将二维示例中的 `k.T` 推广到实际四维张量时，只需交换最后两个轴。实际 key 的形状为 `[B,H,N,D/H]`，应转换为 `[B,H,D/H,N]`，同时保留 batch 和 head 轴。

## 数值检查

运行完整检查：

```bash
python check.py --stage all
```

项目还支持分阶段检查：

```bash
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

检查按形状、有限值和 `abs(actual-reference) <= atol + rtol*abs(reference)` 判断。预处理容差为 `atol=2e-6`；内部 tokens 的容差为 `atol=2e-4, rtol=2e-4`；输出向量的容差为 `atol=2e-5, rtol=2e-4`。程序会报告最大和平均绝对误差。批次检查会改变输入顺序并重复图像。

检查使用的中间参考数组只用于定位差异，不能作为编码器的输入或缓存输出。实现还应支持其他图像和批次大小。

## 模型约定

1. 每个图像块为 `32×32` 像素，图像被切成 `7×7=49` 个块；每个块按 `C,H,W` 展平。
2. patch 投影没有 bias，使用 `patch_weight.T`。
3. 注意力和 MLP 的线性层权重按 `[out,in]` 保存，计算时使用 `weight.T`。
4. `visual.proj` 的形状为 `[768,512]`，已是 `[in,out]`，不要转置。
5. 使用给定的精确 GELU 和 Norm epsilon；不更新任何模型参数。
6. 每个图像 block 包含两条残差连接，不含 LayerScale 参数。

## 提交说明

提交完成的 `image_encoder.py`、本 README 和实验报告。报告应记录各阶段数值误差，并说明一次形状变化、一个修复的问题，以及代码贡献和所用协助。不要重复提交较大的模型参数文件或参考数据。报告正文共 3 页，封面不计入页数。

## 在最终项目中复用

将本项目完成的 `image_encoder.py` 和提供的 `helpers.py` 复制到单独发布的最终项目附件中，与 `clip.py`、`text_encoder.py` 放在同一目录。`ImageEncoder(weights)`、`LayerNorm` 和 `TransformerBlock(weights,prefix,heads)` 保持所需接口。可选的 causal 开关用于兼容后续文本编码器。

最终项目仍须使用其完整的图像/文本检查点和 tokenizer。不要把本项目的图像专用权重当作完整多模态检查点。
