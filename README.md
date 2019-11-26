# omnigan


## interfaces

### batches
```python
batch = {
    "d": depthmap,
    "h": heightmap,
    "w": water_segmentation_map,
    "s": segmentation_map,
    "x": real_flooded_image,
}
```

### data

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
or
```
[
    {
        "x": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005.jpg",
        "s": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005.npy",
        "d": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000005_depth.jpg"
    },
    {
        "x": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006.jpg",
        "s": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006.npy",
        "d": "/Users/victor/Documents/ccai/github/omnigan/example_data/gsv_000006_depth.jpg"
    }
]
```

loaders

```
loaders = Dict({
    train: { rn: loader, rf: loader, sn: loader, sf: loader},
    val: { rn: loader, rf: loader, sn: loader, sf: loader}
})
```
