
from huggingface_hub import snapshot_download

# Scarica i checkpoint (se non già scaricati)
snapshot_download(repo_id="zhengchong/CatVTON")
snapshot_download(repo_id="booksforcharlie/stable-diffusion-inpainting")