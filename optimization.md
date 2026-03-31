# Hunyuan3D 2.1 Performance Optimization

RTX 5090 (32 GB), PyTorch 2.9.1+cu128, test image: `test_character.png`

## Summary

| | Before | After | Speedup |
|---|---|---|---|
| Shape generation | 24.1s | 6.8s | 3.5x |
| Texture generation | 36.2s | 28.9s | 1.25x |
| **End-to-end** | **60.3s** | **~36s** | **~1.7x** |

## Shape Generation (24.1s -> 6.8s)

### Where the time went

| Phase | Time |
|---|---|
| Diffusion sampling (50 steps, 4.93 it/s) | 10.1s |
| Volume decoding (7134 chunks, 556 it/s) | 12.8s |
| Other (conditioning, export) | 1.2s |

### What was optimized

**FlashVDM** (`pipeline.enable_flashvdm(replace_vae=False, mc_algo='mc')`)

Replaced the `VanillaVolumeDecoder` with `FlashVDMVolumeDecoding` — a hierarchical approach that only queries near-surface regions. Reduces volume decoding from 7134 dense grid chunks to 64 adaptive chunks.

- Volume decoding: 12.8s -> ~1.5s
- Shape generation total: 24.1s -> 12.3s

FlashVDM was built for Hunyuan3D v2.0 (which has a turbo VAE variant). For v2.1, the turbo VAE doesn't exist, but the hierarchical volume decoder still works. We use `replace_vae=False` since the v2.1 VAE mapping isn't in the turbo table. Quality is visually identical.

**Reduced diffusion steps** (50 -> 25)

The flow matching Euler scheduler with `sigmas = np.linspace(0, 1, N)` handles fewer steps gracefully. Visually confirmed that 25 steps produces equivalent quality to 50 steps.

- Diffusion: 10.1s -> ~5.1s
- Shape generation total: 12.3s -> 6.8s

### What didn't work

| Optimization | Result | Why |
|---|---|---|
| bf16 dtype | 1% slower | Pipeline already uses fp16. bf16 has lower mantissa precision, no benefit on this workload. |
| cuDNN benchmark mode | 5% slower | Input sizes vary during processing, causing suboptimal kernel selection. |
| torch.compile (shape) | 3% faster diffusion | Marginal gain. MoE layers in the denoiser likely cause graph breaks. 50s warmup cost. |

## Texture Generation (36.2s -> 28.9s)

### Where the time went

| Phase | Time |
|---|---|
| Remesh (quadric decimation 1M -> 40k faces) | 8.4s |
| UV wrap (xatlas) | 1.6s |
| Trimesh load + pymeshlab conversion | 2.3s |
| View selection + normal/position rendering | 0.6s |
| Multiview diffusion (15 steps UniPC, 6 views) | 7.5s |
| ESRGAN super-resolution (12 images, 512->2048) | 2.4s |
| Texture baking (back_sample, 2x for albedo+MR) | 2.2s |
| Inpaint (meshVerticeInpaint + cv2.inpaint at 4096x4096) | 9.3s |
| Save mesh | 0.5s |

### What was optimized

**Reduced views** (6 -> 4)

Discovered that `max_selected_view_num=4` in the original code still selected 6 views because `bake_view_selection` always picked the first 6 candidates before the greedy loop. Fixed `pipeline_utils.py` to respect the cap: `range(min(6, max_selected_view_num))`.

With 4 actual views, the diffusion UNet batch drops from `3 * 2 * 6 = 36` to `3 * 2 * 4 = 24` latents per step (33% less compute). ESRGAN calls drop from 12 to 8. Baking iterations drop proportionally.

- Diffusion: 7.5s -> 4.6s
- ESRGAN: 2.4s -> 1.6s
- Baking: 2.2s -> 1.6s

**Reduced diffusion steps** (15 -> 8)

UniPC is a second-order multistep scheduler that converges quickly. 8 steps confirmed visually equivalent to 15.

- Diffusion: 4.6s -> 2.8s

### What didn't work

| Optimization | Result | Why |
|---|---|---|
| Skip ESRGAN | Blurry textures | The 4x upscale adds critical high-frequency detail to albedo textures. |
| Skip vertex inpaint | No speedup | meshVerticeInpaint takes only 0.09s. The 9.3s is dominated by cv2.inpaint. |
| Lower texture_size (4096->2048) | Blurry textures | Inpaint drops from 9.3s to 2.4s, but output quality is unacceptable. |
| Downsample before inpaint | Corrupted output | set_texture/save_mesh expect texture matching renderer's internal texture_size. |
| pyfqmr fast remesh | Lower mesh quality | 10x faster decimation (0.85s vs 8.4s) but visibly worse mesh topology. Available as opt-in `fast_remesh=true`. |

### Remaining bottlenecks

Both are CPU-bound and tied to quality-critical parameters:

- **Quadric decimation** (8.4s) — reducing 1M faces to 40k. No faster alternative without visible quality loss.
- **cv2.inpaint Navier-Stokes** (8.5s) — operating on 4096x4096 texture. Can't lower resolution without blurriness.

## Quantization Study (fp8 / nvfp4 / int4)

Tested whether reduced-precision quantization could speed up the GPU diffusion phases further.

| Config | Shape gen | Speedup | Warmup | Status |
|---|---|---|---|---|
| fp16 baseline | 6.73s | 1.00x | — | Current default |
| fp8 dynamic (no compile) | 6.73s | 1.00x | — | No actual fp8 compute without torch.compile |
| fp8 dynamic + compile | 5.76s | 1.17x | 134s | Works, ~1s gain |
| int4 weight (torchao) | — | — | — | Requires `mslk` library (placeholder on PyPI, not publicly available) |
| nvfp4 (Blackwell native) | — | — | — | Not available in torchao. Requires PyTorch 2.12+ and unreleased libraries |
| PyTorch 2.12 nightly | — | — | — | fp8+compile hits torch.dynamo bug; int4 still blocked on mslk |

**Conclusion:** fp8+compile saves ~1s with a 2+ minute one-time warmup — not practical unless processing many images per session. nvfp4 and int4 quantization ecosystems aren't mature enough yet (as of March 2026). Worth revisiting when PyTorch 2.12 goes stable and `mslk` gets a real release.

The GPU diffusion phases (5s shape + 3s texture = 8s) are no longer the bottleneck — the 17s of CPU work (remesh 8.4s + cv2.inpaint 8.5s) is. Quantization addresses the wrong bottleneck.

## API Configuration (`/convert-image-to-3d`)

All optimizations are exposed as query parameters with tuned defaults:

| Parameter | Default | Description |
|---|---|---|
| `steps` | 25 | Shape diffusion steps (original: 50) |
| `texture_steps` | 8 | Texture diffusion steps (original: 15) |
| `texture_views` | 4 | Number of texture views (original: 6, was silently using 6 even when set to 4) |
| `smooth_normals` | false | Apply smooth vertex normals to output GLB |
| `fast_remesh` | false | Use pyfqmr for faster but lower quality mesh decimation |
| `cancel_previous` | false | Cancel any in-progress generation |

Example:

```
POST /convert-image-to-3d?steps=25&texture_steps=8&texture_views=4&smooth_normals=true
```

FlashVDM is always enabled on the shape pipeline (no per-request toggle needed — it's purely faster with no quality impact).

## Changes made

| File | Change |
|---|---|
| `demo.py` | FlashVDM enabled, 25 shape steps |
| `api.py` | FlashVDM enabled, per-request `steps`/`texture_steps`/`texture_views`/`fast_remesh` params, per-phase timing logs |
| `hy3dpaint/textureGenPipeline.py` | `texture_steps` and `fast_remesh` config, per-phase timing logs |
| `hy3dpaint/utils/multiview_utils.py` | `num_inference_steps` override support |
| `hy3dpaint/utils/pipeline_utils.py` | Fixed view count to respect `max_selected_view_num` cap |
| `hy3dpaint/utils/simplify_mesh_utils.py` | Added pyfqmr fast decimation path |
