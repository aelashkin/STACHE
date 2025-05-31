import os
import math
from PIL import Image, ImageDraw

from stache.envs.minigrid.state_utils import get_agent_position


def rotate_point(point, center, angle_degrees):
    """
    Rotate a point around a center by angle in degrees.
    """
    angle = math.radians(angle_degrees)
    px, py = point
    cx, cy = center
    qx = cx + math.cos(angle) * (px - cx) - math.sin(angle) * (py - cy)
    qy = cy + math.sin(angle) * (px - cx) + math.cos(angle) * (py - cy)
    return (qx, qy)


def visualize_robustness_region_maps(robustness_region, env, output_dir='rr_maps'):
    """
    Generate and save aggregated maps of the robustness region, one image per agent direction.

    Args:
        robustness_region (list): List of symbolic states (each with 'direction' and agent 'objects').
        env (gym.Env): A MiniGrid environment compatible with full observation wrapper.
        output_dir (str): Directory in which to save the aggregated map images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine tile size (default to 32 if not present)
    tile_size = getattr(env.unwrapped, 'tile_size', 32)
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height

    # Render base grid without agent or highlights
    base_img = grid.render(tile_size, agent_pos=(-1, -1), agent_dir=None)
    base_pil = Image.fromarray(base_img)

    # Group agent positions by direction
    dir_positions = {d: [] for d in range(4)}
    for state in robustness_region:
        d = state.get('direction')
        pos = get_agent_position(state)
        if d is None or pos is None:
            continue
        dir_positions[d].append(pos)

    # Draw aggregated maps per direction
    for d, positions in dir_positions.items():
        if not positions:
            continue
        img = base_pil.copy()
        draw = ImageDraw.Draw(img)
        # triangle half-size
        r = tile_size // 4
        for x, y in positions:
            # pixel center for the cell
            cx = x * tile_size + tile_size // 2
            cy = y * tile_size + tile_size // 2
            # define upward-pointing triangle
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            # rotate according to direction
            rotated = [rotate_point(pt, (cx, cy), d * 90) for pt in pts]
            draw.polygon(rotated, fill=(255, 0, 0))
        # save image
        filename = f'dir_{d}.png'
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
