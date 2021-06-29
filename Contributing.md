1. Understand the file structure:
   1. architectures in `discriminator.py` `generator.py` `classifier.py`
   2. data loading in `data.py`
   3. data transformation `transforms.py`
   4. optimizers in `optim.py`
   5. utilities in `utils.py`
   6. training procedure in `trainer.py`
2. Write **tests** in `tests/`
   1. your file should match `test_*.py`
   2. update existing tests when adding functionalities
   3. run tests regularly to check you haven't broken anything `python tests/run.py`
3. Add **WIP** in your PR's title when not ready to merge
5. Open an Issue if something's odd, or to assign yourself a todo
6. **Format your code** with [black](https://github.com/psf/black)
7. Only update `trainer/defaults.yaml` with values that should be shared across runs and users
   1. use `config/trainer/local_tests.yaml` or any other to setup your particular config overriding `trainer/defaults.yaml`

## Running tests

As per `7.` you should set your particular config in `config/local_tests.yaml`. Mine looks like:

```yaml
output_path: /Users/victor/Documents/ccai/github/climategan/example_data
# -------------------
# -----  Tasks  -----
# -------------------
#tasks: [a, d, h, s, t, w]
tasks: [a, d, s, t] # for now no h or w
# ----------------
# ----- Data -----
# ----------------
data:
  files: # if one is not none it will override the dirs location
    base: /Users/victor/Documents/ccai/github/climategan/example_data
  transforms:
  - name: hflip
    ignore: false
    p: 0.5
  - name: resize
    ignore: false
    new_size: 256
  - name: crop
    ignore: false
    height: 64
    width: 64
gen:
  encoder:
    n_res: 1
  default:
    n_res: 1

train:
  log_level: 1
```

Setting `n_res` to 1 is important to run tests faster and with less memory
