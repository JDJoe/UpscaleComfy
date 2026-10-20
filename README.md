# UpscaleComfy

Upscales every image in the current directory with a ComfyUI workflow.

ComfyUI has to be running. The script uploads each image, queues the workflow on `http://127.0.0.1:8188`, and waits for it to finish. ComfyUI saves inside its own output folder. The script then moves each result into `./upscaled/` and names it after the source file. An absolute Save Image prefix is rewritten so the save stays inside that output folder.

The workflow file must be the API format, with a `class_type` on each node. It needs a Load Image node and a Save Image node. `SeedVR upscale-0.json` is the included SeedVR workflow.

From the folder that holds the pictures:

```bash
python /path/to/upscale.py "/path/to/SeedVR upscale-0.json"
```

Images already in `./upscaled/` are left alone. The script only reads pictures in the current directory: png, jpg, jpeg, webp, bmp, gif, tif, and tiff.
