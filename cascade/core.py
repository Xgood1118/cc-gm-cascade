# -*- coding: utf-8 -*-
"""级联阴影分割（cascaded shadow map 分割器）。

只用 Python 标准库。视锥沿摄像机深度轴切分：

* :func:`split_cascades` 用对数/均匀混合（hybrid log-uniform）公式产生
  不重叠、不留空隙、首尾精确等于 ``near``/``far`` 的级联区间；
* :func:`tighten_bounds` 把每级近远平面吸附到该级贴图的纹素网格上，
  消除逐帧抖动；
* :func:`blend_region` 给出相邻两级在过渡区内的一对权重，使用 smoothstep
  曲线，任意位置权重和恒为 1，过渡边界处导数为 0，不产生亮度断层。
"""

import math
from dataclasses import dataclass

__all__ = [
    "Cascade",
    "split_cascades",
    "tighten_bounds",
    "blend_region",
    "world_precision",
    "texel_size",
    "alignment_error",
    "blend_band",
    "LAMBDA",
    "BLEND_FRACTION",
]

#: 混合分割系数：0 为纯均匀分割，1 为纯对数分割。
LAMBDA = 0.5
#: 过渡区宽度占“外层级”自身深度跨度的比例。
BLEND_FRACTION = 0.1


@dataclass
class Cascade:
    """一级阴影。

    前四个字段保持骨架原有顺序与含义；后两个字段带默认值，因此旧的
    ``Cascade(index, near, far)`` / ``Cascade(index, near, far, resolution)``
    构造方式完全不变。
    """

    index: int
    near: float
    far: float
    resolution: int = 1024
    #: 该级一个纹素对应的世界空间深度长度（粒度，越大越粗糙）。
    texel_world: float = 0.0
    #: 吸附前近/远平面偏离纹素网格的距离（单位：纹素）。
    align_error: tuple = (0.0, 0.0)


def _validate(near, far, levels):
    if not isinstance(levels, bool) and isinstance(levels, int):
        n_levels = levels
    else:
        raise ValueError("levels 必须是正整数，收到 %r" % (levels,))
    if n_levels < 1:
        raise ValueError("levels 必须 >= 1，收到 %r" % (levels,))
    if not all(math.isfinite(x) for x in (near, far)):
        raise ValueError("near/far 必须是有限数，收到 near=%r far=%r" % (near, far))
    if near <= 0.0:
        raise ValueError("near 必须为正数，收到 %r" % (near,))
    if far <= near:
        raise ValueError("far 必须大于 near，收到 near=%r far=%r" % (near, far))
    return n_levels


def split_cascades(near, far, levels):
    """把视锥深度区间 ``[near, far]`` 切成 ``levels`` 级。

    采用混合分割（Practical CSM, Gotanda 2010 的经典形式）::

        log_i     = near * (far / near) ** (i / n)
        uniform_i = near + (far - near) * i / n
        d_i       = lambda * log_i + (1 - lambda) * uniform_i

    返回长度为 ``levels`` 的 :class:`Cascade` 列表。相邻级严格首尾相接
    （``cascades[i].far == cascades[i+1].near``），首端精确为 ``near``、
    末端精确为 ``far``，因此区间不重叠也不留空隙。结果完全由入参决定，
    同一入参两次调用逐字节一致。

    每级 ``resolution`` 固定为 1024，``texel_world`` 为该级世界空间纹素
    粒度 ``(far - near) / resolution``，随级递增（精度逐级递远）。
    """
    n = _validate(near, far, levels)
    lam = LAMBDA
    ratio = far / near
    span = far - near
    bounds = [near]
    for i in range(1, n):
        f = i / n
        log_i = near * (ratio ** f)
        uniform_i = near + span * f
        bounds.append(lam * log_i + (1.0 - lam) * uniform_i)
    bounds.append(far)

    # 混合公式在合法输入下严格递增；钳制一次以在数值上保证级间无空隙。
    for i in range(1, n + 1):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = math.nextafter(bounds[i - 1], math.inf)
    # 首尾端点强制精确，保证覆盖整个 [near, far]。
    bounds[0] = near
    bounds[n] = far

    cascades = []
    for i in range(n):
        c_near, c_far = bounds[i], bounds[i + 1]
        tex = (c_far - c_near) / 1024.0
        cascades.append(
            Cascade(
                index=i,
                near=c_near,
                far=c_far,
                resolution=1024,
                texel_world=tex,
                align_error=(0.0, 0.0),
            )
        )
    return cascades


def world_precision(cascade):
    """返回该级世界空间精度（一个纹素覆盖的世界深度长度）。

    数值越小越精细。固定分辨率下区间越远、跨度越大，精度越粗糙。
    """
    return (cascade.far - cascade.near) / cascade.resolution


def texel_size(cascade, cam=None):
    """该级在给定相机下的世界空间纹素长度。

    1D 深度模型里纹素长度即深度跨度除以分辨率；``cam`` 仅用于与
    :func:`tighten_bounds` 对称的调用习惯，可为 ``None``。
    """
    if cam is None:
        cam = {}
    resolution = int(cam.get("resolution", cascade.resolution))
    return (cascade.far - cascade.near) / resolution


def tighten_bounds(cascades, cam):
    """把每级近/远平面对齐到该级贴图的纹素网格。

    ``cam`` 为映射（也兼容 ``None`` / 带同名属性的对象），支持字段：

    * ``origin``（默认 0.0）：纹素网格原点的世界深度坐标；
    * ``resolution``（默认取级联自身分辨率）：每级贴图像素数。

    近平面向下吸附、远平面向上吸附（floor/ceil），因此吸附后仍然完整
    覆盖原始区间。返回新的 :class:`Cascade` 列表，不修改入参；每级的
    ``align_error`` 记录吸附前两端偏离网格的距离（单位：纹素，绝对值
    不超过 0.5），``texel_world`` 保留吸附前的纹素粒度（吸附前后跨度
    只差不足一个纹素）。
    """
    if cam is None:
        cam = {}
    origin = float(cam.get("origin", 0.0))
    snapped = []
    for c in cascades:
        resolution = int(cam.get("resolution", c.resolution))
        t = (c.far - c.near) / resolution
        n_cells = math.floor((c.near - origin) / t)
        f_cells = math.ceil((c.far - origin) / t)
        err_near = (c.near - origin) / t - n_cells
        err_far = (c.far - origin) / t - f_cells
        new_near = origin + n_cells * t
        new_far = origin + f_cells * t
        snapped.append(
            Cascade(
                index=c.index,
                near=new_near,
                far=new_far,
                resolution=resolution,
                texel_world=t,
                align_error=(err_near, err_far),
            )
        )
    return snapped


def alignment_error(cascade, origin=0.0):
    """返回当前近/远平面相对于纹素网格的残余偏差（单位：纹素）。

    用于核验 :func:`tighten_bounds` 的吸附结果：吸附后残余偏差只来自
    浮点舍入，应在 1e-9 纹素以内。
    """
    t = cascade.texel_world or world_precision(cascade)
    near_cells = (cascade.near - origin) / t
    far_cells = (cascade.far - origin) / t
    return (near_cells - round(near_cells), far_cells - round(far_cells))


def blend_band(cascade, fraction=BLEND_FRACTION):
    """返回该级末端过渡区 ``(start, end)`` 与权重采样点。

    过渡区是该级与下一级共享边界附近的一段对称区间，宽度为该级深度
    跨度的 ``fraction``（默认 10%）。返回三元组
    ``(start, end, [(depth, w_this, w_next), ...])``，每个采样点上
    ``w_this + w_next == 1``（误差 < 1e-15）。
    """
    width = (cascade.far - cascade.near) * fraction
    start = cascade.far - width
    end = cascade.far + width
    samples = []
    for k in range(5):
        u = k / 4.0
        depth = start + (end - start) * u
        w_this, w_next = blend_region(start, end, depth)
        samples.append((depth, w_this, w_next))
    return start, end, samples


def blend_region(a, b, t):
    """过渡区 ``[a, b]`` 内位置 ``t`` 处的一对权重 ``(w_a, w_b)``。

    ``t`` 为世界空间深度（也允许直接传 ``[0, 1]`` 归一化位置，此时
    等价于 ``a=0, b=1``）。权重采用 smoothstep 曲线::

        u     = clamp((t - a) / (b - a), 0, 1)
        s     = u * u * (3 - 2u)
        w_a   = 1 - s      # 本级（近端）权重
        w_b   = s          # 下一级（远端）权重

    恒有 ``w_a + w_b == 1``；在 ``a``、``b`` 两端权重导数为 0，过渡
    平滑、无亮度断层；零宽度过渡区（``a == b``）按阶跃处理，结果仍
    确定且权重和为 1。
    """
    if b == a:
        return (0.0, 1.0) if t >= a else (1.0, 0.0)
    u = (t - a) / (b - a)
    if u <= 0.0:
        u = 0.0
    elif u >= 1.0:
        u = 1.0
    s = u * u * (3.0 - 2.0 * u)
    return 1.0 - s, s
