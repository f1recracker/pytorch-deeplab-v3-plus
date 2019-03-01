
import torch

def bdd_palette(seg_labels):
    color_map = {
        0: torch.FloatTensor((128, 67, 125)), # Road
        1: torch.FloatTensor((247, 48, 227)), # Sidewalk
        2: torch.FloatTensor((72, 72, 72)), # Building
        3: torch.FloatTensor((101, 103, 153)), # Wall
        4: torch.FloatTensor((190, 151, 152)), # Fence
        5: torch.FloatTensor((152, 152, 152)), # Pole
        6: torch.FloatTensor((254, 167, 56)), # Light
        7: torch.FloatTensor((221, 217, 55)), # Sign
        8: torch.FloatTensor((106, 140, 51)), # Vegetation
        9: torch.FloatTensor((146, 250, 157)), # Terrain
        10: torch.FloatTensor((65, 130, 176)), # Sky
        11: torch.FloatTensor((224, 20, 64)), # Person
        12: torch.FloatTensor((255, 0, 25)), # Rider
        13: torch.FloatTensor((0, 22, 138)), # Car
        14: torch.FloatTensor((0, 11, 70)), # Truck
        15: torch.FloatTensor((0, 63, 98)), # Bus
        16: torch.FloatTensor((0, 82, 99)), # Train
        17: torch.FloatTensor((0, 36, 224)), # Motorcycle
        18: torch.FloatTensor((121, 17, 38)), # Bicycle
    }
    seg_colors = torch.zeros_like(seg_labels).repeat(3, 1, 1).float()
    for c_id, color in color_map.items():
        color = color.to(seg_labels.device).float() / 255
        seg_colors += (seg_labels == c_id).float() * color [:, None, None]
    return seg_colors
