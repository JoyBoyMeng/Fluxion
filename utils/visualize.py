# utils/fluxon_once_vis.py
from typing import Optional, Tuple, Dict
from pathlib import Path
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib import colors as mcolors

matplotlib.set_loglevel("warning")

# 模块级缓存：按 (D, seed) 记忆正交基，避免重复计算
_BASIS_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}

def _load_basis_from_file(basis_file: Path) -> Optional[torch.Tensor]:
    if basis_file.exists():
        obj = torch.load(basis_file, map_location="cpu")
        if isinstance(obj, torch.Tensor) and obj.ndim == 2 and obj.shape[1] == 3:
            return obj
    return None

def _save_basis_to_file(basis: torch.Tensor, basis_file: Path) -> None:
    basis_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis.cpu(), basis_file)

# 在文件里新增一个小工具函数（放在 visualize_fluxons 上面即可）：
def _fixed_row_colors(N: int, sat: float = 0.65, val: float = 0.9) -> np.ndarray:
    """
    生成稳定的行索引调色板：HSV 中等饱和/亮度，N 个均匀色相。
    只要 N 不变，行 i 的颜色在每次调用都一致。
    返回 shape [N, 3] 的 RGB 数组（0~1）。
    """
    if N <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    hues = np.linspace(0.0, 1.0, N, endpoint=False)
    hsv = np.stack([hues, np.full(N, sat), np.full(N, val)], axis=1)
    rgb = mcolors.hsv_to_rgb(hsv).astype(np.float32)
    return rgb

def _get_fixed_basis(D: int, seed: int, device: torch.device,
                     basis_file: Optional[str]) -> torch.Tensor:
    """
    获得固定正交基 Q ∈ R^{D×3}。
    优先：内存缓存 -> 磁盘缓存 -> 生成（QR）
    """
    key = (D, seed)
    if key in _BASIS_CACHE:
        # print('可视化--读取旧基')
        return _BASIS_CACHE[key].to(device)

    if basis_file is not None:
        bf = Path(basis_file)
        loaded = _load_basis_from_file(bf)
        if loaded is not None:
            # print('可视化--读取旧基')
            # 若磁盘基维度或种子不匹配，仍以此为准（只要列数=3）
            _BASIS_CACHE[key] = loaded.to(device)
            return _BASIS_CACHE[key]

    # 生成新基（固定 seed，QR 正交化）
    # print('可视化--生成新基')
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    M = torch.randn(D, 3, generator=g, device=device)
    Q, _ = torch.linalg.qr(M, mode="reduced")  # [D, 3]
    _BASIS_CACHE[key] = Q

    if basis_file is not None:
        _save_basis_to_file(Q, Path(basis_file))
    return Q

def visualize_fluxons(
    vectors: torch.Tensor,
    save_path: str = "fluxons.jpg",
    *,
    seed: int = 42,                  # 固定投影基的随机种子（保证一致）
    basis_file: Optional[str] = None,# 可选：基落盘路径（跨进程/重启也一致），如 ".cache/fluxon_basis_D128_seed42.pt"
    normalize: bool = False,         # False 展示真实“扩散”；True 显示球面分布
    color_by: str = "norm",          # "norm" | "none"
    point_size: float = 7.0,
    point_alpha: float = 0.9,
    elev: float = 22.0,
    azim: float = 45.0,
    dpi: int = 300,
    show_wireframe: bool = True,
    title: Optional[str] = None,
) -> torch.Tensor:
    """
    单次可视化一个 [N, D] 张量，使用固定基投影到 3D 并保存 JPG。
    返回用于投影的基 Q (D×3)，方便你显式保存/复用。

    使用建议：
      - 训练过程中每隔若干 batch 调用，用相同 seed/basis_file，输出不同图片文件名。
      - 若你需要跨脚本/重启保持一致基，提供 basis_file 路径。
    """
    if not isinstance(vectors, torch.Tensor):
        raise TypeError(f"`vectors` must be torch.Tensor, got {type(vectors)}")
    if vectors.ndim != 2:
        raise ValueError(f"`vectors` must be 2D [N, D], got {tuple(vectors.shape)}")

    N, D = vectors.shape
    if D < 3:
        raise ValueError(f"D must be >= 3 for 3D projection, got D={D}")

    device = vectors.device
    Q = _get_fixed_basis(D, seed, device, basis_file)   # [D, 3]

    X = vectors.detach()

    # 2) 投影
    P = (X @ Q).to("cpu")  # [N, 3]

    # 3) 颜色
    if color_by == "norm":
        c = X.norm(dim=1).to("cpu").numpy()
        cmap = "viridis"
        vmin = None
        vmax = None
    elif color_by == "row":
        # 稳定：按行索引给定固定 RGB 颜色，不依赖数据取值
        c = _fixed_row_colors(N)  # [N, 3] RGB
        cmap = None
        vmin = None
        vmax = None
    else:
        c = None
        cmap = None
        vmin = None
        vmax = None

    # 4) 坐标轴范围（保持等比例立方体）
    max_abs = float(P.abs().max())
    pad = 0.05 * max_abs if max_abs > 0 else 0.5
    lo, hi = -max_abs - pad, max_abs + pad

    # 5) 绘图
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        P[:, 0].numpy(), P[:, 1].numpy(), P[:, 2].numpy(),
        s=point_size, alpha=point_alpha,
        c=c, cmap=cmap if c is not None else None,
        edgecolors="white", linewidth=0.25
    )
    if show_wireframe:
        u = torch.linspace(0, 2 * torch.pi, 48)
        v = torch.linspace(0, torch.pi, 24)
        cu, su = torch.cos(u), torch.sin(u)
        sv, cv = torch.sin(v), torch.cos(v)
        Xw = torch.outer(cu, sv)
        Yw = torch.outer(su, sv)
        Zw = torch.outer(torch.ones_like(u), cv)
        ax.plot_wireframe(
            Xw.numpy(), Yw.numpy(), Zw.numpy(),
            color="lightgray", linewidth=0.25, alpha=0.3, rstride=2, cstride=2
        )

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=12, pad=8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return Q
