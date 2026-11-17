# 级联阴影分割

一个把摄像机视锥切成多级阴影贴图、并保证级间衔接无缝的分割器

## 保留签名

- `Cascade`
- `split_cascades(near, far, levels)`
- `tighten_bounds(cascades, cam)`
- `blend_region(a, b, t)`

## 自测

```
python -m pytest tests -q
python -m cascade.cli data/split.json --levels 4 --out cascades.json
```
