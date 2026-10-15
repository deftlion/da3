# Depth Anything 3 model-building vendor subset

Source: ByteDance-Seed/Depth-Anything-3, local checkout commit `c992b784c60608e191667e036910fe011566d2a8`. The listed source files and six YAML presets are copied from `src/depth_anything_3/` at that commit; `api.py` is trimmed to model construction and raw forward, and `__init__.py` is a local package marker. Original copyright headers remain.

License: Apache-2.0; see `LICENSE`. DINOv2-derived layers retain their Meta copyright/license notice. The upstream full package restricts numpy to <2 and includes inference/export dependencies; this subset does not contain the inference processor, export, or CLI. Gaussian rendering is not part of this Space quantization path.
