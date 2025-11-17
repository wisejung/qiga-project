import networkx as nx
import numpy as np

import json

def save_pos(pos, filename="pos.json"):
    pos_serializable = {str(k): v.tolist() for k, v in pos.items()}
    with open(filename, "w") as f:
        json.dump(pos_serializable, f)


def load_pos(filename="pos.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}
# nx.draw(G2, pos=pos_loaded, with_labels=True)


from PIL import Image
import math

def merge_images_grid(image_paths, grid_rows, grid_cols, output_path="merged.png", 
                      padding=10, bg_color=(255, 255, 255)):

    images = [Image.open(p) for p in image_paths]

    w = max(img.width for img in images)
    h = max(img.height for img in images)

    total_width = grid_cols * w + padding * (grid_cols + 1)
    total_height = grid_rows * h + padding * (grid_rows + 1)

    merged = Image.new("RGB", (total_width, total_height), color=bg_color)

    for idx, img in enumerate(images):
        if idx >= grid_rows * grid_cols:
            break

        row = idx // grid_cols
        col = idx % grid_cols

        x = padding + col * (w + padding)
        y = padding + row * (h + padding)

        merged.paste(img, (x, y))

    merged.save(output_path)
    print(f"Saved merged image at {output_path}")