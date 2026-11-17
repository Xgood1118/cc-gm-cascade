# -*- coding: utf-8 -*-
"""级联阴影分割命令行。

用法::

    python -m cascade.cli data/split.json --levels 4 --out cascades.json
"""

import argparse
import json

import cascade as m


def build_payload(cfg, levels):
    """根据配置与级数构造可 JSON 序列化的结果字典（纯函数，两次调用一致）。"""
    near = float(cfg["near"])
    far = float(cfg["far"])
    resolution = int(cfg.get("resolution", 1024))
    cam = dict(cfg.get("cam") or {})
    cam.setdefault("resolution", resolution)

    cascades = m.split_cascades(near, far, levels)
    snapped = m.tighten_bounds(cascades, cam)

    out_cascades = []
    for c, s in zip(cascades, snapped):
        start, end, samples = m.blend_band(c)
        residual = m.alignment_error(s, origin=float(cam.get("origin", 0.0)))
        out_cascades.append(
            {
                "index": s.index,
                "near": s.near,
                "far": s.far,
                "resolution": s.resolution,
                "world_precision": m.world_precision(c),
                "texel_world": s.texel_world,
                "blend": {
                    "start": start,
                    "end": end,
                    "samples": [
                        {
                            "depth": d,
                            "weight_this": wa,
                            "weight_next": wb,
                            "weight_sum": wa + wb,
                        }
                        for d, wa, wb in samples
                    ],
                },
                "snap_error_texels": {
                    "near_before": s.align_error[0],
                    "far_before": s.align_error[1],
                    "near_residual": residual[0],
                    "far_residual": residual[1],
                },
            }
        )

    return {
        "near": near,
        "far": far,
        "levels": levels,
        "lambda": m.LAMBDA,
        "blend_fraction": m.BLEND_FRACTION,
        "cascades": out_cascades,
    }


def main():
    ap = argparse.ArgumentParser(description="级联阴影分割")
    ap.add_argument("input", help="相机配置 JSON，例如 data/split.json")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.input, encoding="utf-8") as f:
        cfg = json.load(f)

    payload = build_payload(cfg, a.levels)
    # sort_keys + 固定分隔符，保证同样输入两次输出逐字节一致。
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(
        "wrote %d cascades to %s" % (len(payload["cascades"]), a.out)
    )


if __name__ == "__main__":
    main()
