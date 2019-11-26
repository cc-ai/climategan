# omnigan


## interfaces

```python
batch = {
    "d": depthmap,
    "h": heightmap,
    "w": water_segmentation_map,
    "s": segmentation_map,
    "x": real_flooded_image,
    "domain": rf | rn | sf | sn
}
```

```yaml
# data file ; one for each r|s-f|n
- x: /path/to/image
  h: /path/to/height map
  d: /path/to/depth map
  w: /path/to/water map
  s: /path/to/segmentation map
- x: /path/to/another image
  d: /path/to/depth map
  s: /path/to/segmentation map
- x: ...
```
