import torch
from einops import rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_chans=3, embed_dim=768
    ):
        super().__init__()
        """
        """
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.projection = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.positions = nn.Parameter(
            torch.rand(1, (img_size[0] // patch_size[0]) ** 2 + 1, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # проверка на размер изображения
        b, c, h, w = x.shape

        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, "1 n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.seq = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(p=drop),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p=drop),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0.0, out_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**0.5

        self.qkv = nn.Sequential(
            nn.Linear(dim, 3 * dim, bias=qkv_bias),
            Rearrange("b n (qkv h e) -> b qkv h n e", qkv=3, h=num_heads),
        )
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.out = nn.Sequential(Rearrange("b h n e -> b n (h e)"), nn.Linear(dim, dim))
        self.out_drop = nn.Dropout(p=out_drop)

    def forward(self, x: Tensor):
        # Attention
        qkv = self.qkv(x)
        q, k, v = unpack(qkv, [[], [], []], "b * h n e")
        k_t = rearrange(k, "b h n e -> b h e n")
        attn = nn.functional.softmax((q @ k_t) / self.scale, dim=-1)
        attn = self.attn_drop(attn)

        # Out projection
        x = self.out(attn @ v)
        x = self.out_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, drop_rate=0.0):
        super().__init__()

        # Attention
        self.attention = Attention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=drop_rate, out_drop=drop_rate
        )

        # Normalization
        self.norm1 = nn.LayerNorm(normalized_shape=(dim))
        self.norm2 = nn.LayerNorm(normalized_shape=(dim))

        # MLP
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop_rate)

    def forward(self, x):
        # Attetnion
        x = self.norm1(x)
        x = self.attention(x) + x
        # MLP
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self, depth, dim, num_heads=8, qkv_bias=False, mlp_ratio=4, drop_rate=0.0
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, qkv_bias, mlp_ratio, drop_rate)
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
    ):
        super().__init__()
        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)

        self.patch_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder
        self.transformer = Transformer(
            depth, embed_dim, num_heads, qkv_bias, mlp_ratio, drop_rate
        )

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor):
        # Path Embeddings, CLS Token, Position Encoding
        x = self.patch_embedding(x)
        x = self.patch_drop(x)

        # Transformer Encoder
        x = self.transformer(x)

        # Classifier
        x = self.classifier(x[:, 0, :])

        return x
