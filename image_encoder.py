"""Student implementation of the final project's image encoder.

Completed NumPy forward calculation for the six project exercises. Model
parameters and numerical helper functions are supplied separately.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02：Split [B,N,D] into [B,H,N,D/H], preserving token order.

        In this image model D=768, H=12 and D/H=64. Reshape separates heads;
        transpose places each head before the token axis.
        """
        batch, token_count, _ = x.shape
        # 先把特征维切成 heads 个 head_dim，再交换轴，使 head 维位于 token 维之前。
        # 这样每个 head 都保留完整的 token 序列，结果形状为 [B,H,N,D/H]。
        return x.reshape(batch, token_count, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].

        # 对每个 head 的 query 和 key 做点积；只交换 k 的最后两个轴，
        # 因而 [B,H,N,Dh] @ [B,H,Dh,N] 得到 [B,H,N,N] 注意力分数。
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].

        # 每个 query 按注意力系数对 value 序列加权求和，得到每个 head 的上下文。
        context = coefficients @ v  # [B,H,N,D/H]
        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.

        # 将 head 轴移回 token 轴之后，再拼接各 head 的特征维。
        merged = context.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.width)

        # 注意力输出权重按 [out,in] 保存，因此线性投影时转置权重并加偏置。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)
        # TODO-04a: first residual addition. Name the result residual.

        # 第一条残差支路保留归一化前的输入 x，而不是 Norm 1 的结果。
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)
        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].

        # 两层 MLP 的权重存储为 [out,in]；GELU 使用题目提供的精确实现。
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        # 第二条残差支路同样保留 MLP 输入 residual。
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """Convert [B,3,224,224] images to [B,49,3072] patch rows.

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        batch = images.shape[0]
        # 将高、宽拆成 7 个 patch 坐标和各 patch 内的 32 个像素坐标。
        patches = images.reshape(batch, 3, 7, 32, 7, 32)
        # 先按从上到下、从左到右排列 patch，再按 C,H,W 展平每个 patch。
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, 49, 3 * 32 * 32)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # Patch 权重原为 [768,3072]；这里转置后映射到 768 维，无额外偏置。
        patch_tokens = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # 在 token 轴上把 CLS 放在 49 个 patch 之前，并广播加上 50 个位置向量。
        tokens = np.concatenate((cls, patch_tokens), axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """Return one normalized 512-value output token per image."""
        tokens = self.encode_tokens(images)

        # 只取每张图像的 CLS 行 [B,768]，依次执行输出归一化和 [768,512] 投影。
        cls_token = tokens[:, 0, :]
        normalized = self.post_norm(cls_token)
        output = normalized @ self.projection  # visual.proj 已是 [in,out]

        return unit_vector(output)
