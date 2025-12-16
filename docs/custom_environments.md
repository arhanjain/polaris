# Creating Custom Environments

The environments we provide were scanned using ZED cameras, but the reconstruction pipeline is camera agnostic. 

Capture a dense view video of a scene without motion blur, and run it through [COLMAP](https://colmap.github.io/install.html)
```bash
sudo apt install colmap ffmpeg

# Split video into frames at desired FPS
ffmpeg -i dense_view.mp4 -vf "fps=10" frames/dense_view_%04d.png
```

