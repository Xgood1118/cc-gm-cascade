# -*- coding: utf-8 -*-
"""级联阴影分割"""

from dataclasses import dataclass


@dataclass
class Cascade:
    """一级阴影。"""
    index: int
    near: float
    far: float
    resolution: int = 1024


def split_cascades(near, far, levels):
    """切分"""
    raise NotImplementedError("split_cascades 尚未实现")


def tighten_bounds(cascades, cam):
    """收紧包围"""
    raise NotImplementedError("tighten_bounds 尚未实现")


def blend_region(a, b, t):
    """过渡区权重"""
    raise NotImplementedError("blend_region 尚未实现")

