# Third-party notices

The majority of this repository is licensed under the Apache License 2.0 (`LICENSE.md`); `README.md` lists the parts
under other licenses. In addition, the following parts of this fork re-implement designs from
[sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) (MIT License); the notice below applies to them:

- `library/anima_block_compile.py` - per-block `torch.compile` of the Anima DiT with a dynamic token axis, token-count
  bands and compile-cache isolation (after anima_lora's `library/anima/models.py` `compile_blocks` /
  `_make_dynamic_seq_forward`, `library/datasets/buckets.py` `cluster_token_bands` / `band_for_seq` and
  `library/runtime/harness.py` `isolate_compile_cache`), re-implemented for this fork's `Block` and LyCORIS modules;
- the fake-5D block loop of `library/anima_models.py` (`Anima.forward_mini_train_dit` under `_block_compile`).

```
MIT License

Copyright (c) 2026 Seunghyun Ji

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
