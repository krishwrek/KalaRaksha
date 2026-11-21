"""
KalaRaksha - Synthetic Heritage Asset Generator
Creates realistic procedural heritage wall images with specific deterioration patterns:
1. sample1_temple_crack.jpg: Granite temple wall with deep structural fracture & hairline fissures
2. sample2_mural_moisture.jpg: Ancient fresco mural with water seepage & capillary damp plume
3. sample3_fresco_peeling.jpg: Lime-plaster wall with severe flaking paint & exposed underlayer
4. sample4_sculpture_moss.jpg: Sandstone bas-relief with green algae & bryophyte moss colonies
5. sample5_sandstone_erosion.jpg: Weathered sandstone with granular pitting & surface erosion
"""

import os
import numpy as np
import cv2
from PIL import Image


def generate_granite_crack_sample(width: int = 960, height: int = 720) -> np.ndarray:
    """Generate ancient granite temple ashlar wall with structural fracture."""
    np.random.seed(42)
    # 1. Base gray granite texture
    base = np.random.normal(160, 20, (height, width)).clip(100, 220).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)

    # 3-channel BGR
    img = cv2.merge([
        (base * 0.92).astype(np.uint8),
        (base * 0.95).astype(np.uint8),
        (base * 0.98).astype(np.uint8)
    ])

    # 2. Add horizontal and vertical ashlar stone mortar joints
    for y in [180, 360, 540]:
        cv2.line(img, (0, y), (width, y), (85, 90, 95), 4)
        cv2.line(img, (0, y+2), (width, y+2), (190, 195, 200), 1)

    for y_start, y_end in [(0, 180), (360, 540)]:
        for x in [300, 650]:
            cv2.line(img, (x, y_start), (x, y_end), (85, 90, 95), 3)
    for y_start, y_end in [(180, 360), (540, 720)]:
        for x in [150, 480, 800]:
            cv2.line(img, (x, y_start), (x, y_end), (85, 90, 95), 3)

    # 3. Draw a prominent structural diagonal fracture with branches
    pts = [
        (220, 40), (260, 120), (310, 210), (380, 300), (430, 410),
        (510, 520), (590, 620), (640, 690)
    ]
    # Add natural jaggedness
    cur_x, cur_y = pts[0]
    for target_x, target_y in pts[1:]:
        steps = 15
        for s in range(1, steps + 1):
            t = s / float(steps)
            nx = int(cur_x + t * (target_x - cur_x) + np.random.randint(-4, 5))
            ny = int(cur_y + t * (target_y - cur_y) + np.random.randint(-3, 4))
            # Deep crack core (dark black/brown)
            cv2.line(img, (cur_x, cur_y), (nx, ny), (25, 28, 32), 4)
            # Frayed spalling border
            cv2.line(img, (cur_x + 1, cur_y + 1), (nx + 1, ny + 1), (50, 55, 60), 2)
            cur_x, cur_y = nx, ny

    # Branch 1
    cur_x, cur_y = 380, 300
    for target_x, target_y in [(490, 340), (580, 365), (660, 380)]:
        for s in range(1, 10):
            t = s / 10.0
            nx = int(cur_x + t * (target_x - cur_x) + np.random.randint(-3, 4))
            ny = int(cur_y + t * (target_y - cur_y) + np.random.randint(-3, 4))
            cv2.line(img, (cur_x, cur_y), (nx, ny), (30, 32, 35), 3)
            cur_x, cur_y = nx, ny

    # Branch 2 (hairline)
    cur_x, cur_y = 510, 520
    for target_x, target_y in [(460, 580), (430, 640)]:
        for s in range(1, 8):
            t = s / 8.0
            nx = int(cur_x + t * (target_x - cur_x) + np.random.randint(-2, 3))
            ny = int(cur_y + t * (target_y - cur_y) + np.random.randint(-2, 3))
            cv2.line(img, (cur_x, cur_y), (nx, ny), (35, 38, 42), 2)
            cur_x, cur_y = nx, ny

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_mural_moisture_sample(width: int = 960, height: int = 720) -> np.ndarray:
    """Generate ancient fresco mural with water seepage and capillary damp plume."""
    np.random.seed(101)
    # 1. Warm plaster fresco base (ochre and cream tones)
    base = np.full((height, width, 3), [195, 215, 230], dtype=np.uint8)  # BGR

    # 2. Add decorative fresco geometric mural motifs (red cinnabar & indigo accents)
    cv2.rectangle(base, (80, 80), (width - 80, height - 80), (120, 140, 200), 6)
    cv2.circle(base, (width // 2, 280), 120, (80, 90, 180), 8)
    cv2.circle(base, (width // 2, 280), 80, (160, 100, 60), 6)

    # Some decorative lotus petal curves
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        cx = int(width // 2 + 100 * np.cos(angle))
        cy = int(280 + 100 * np.sin(angle))
        cv2.circle(base, (cx, cy), 18, (90, 110, 195), -1)

    # 3. Add heavy capillary rising damp / water seepage plume from bottom-center
    # Create smooth organic damp mask
    damp_mask = np.zeros((height, width), dtype=np.float32)

    # Multi-ellipse damp lobes rising upwards
    centers = [
        (480, 650, 260, 180), (360, 580, 180, 150), (600, 600, 190, 140),
        (450, 460, 140, 110), (320, 500, 100, 90), (520, 390, 80, 70)
    ]
    for cx, cy, rx, ry in centers:
        cv2.ellipse(damp_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    # Perturb with noise to make it realistic amoeba-like
    noise = np.random.normal(0, 0.15, (height, width)).astype(np.float32)
    damp_mask = np.clip(damp_mask + noise, 0.0, 1.0)
    damp_mask = cv2.GaussianBlur(damp_mask, (45, 45), 0)

    # Apply moisture color transformation:
    # Dampness darkens Value significantly and introduces brown/ochre mineral stains
    damp_mask_3c = np.repeat(damp_mask[:, :, np.newaxis], 3, axis=2)

    # Damp stain color (dark brownish gray)
    damp_color = np.array([45, 65, 85], dtype=np.float32)  # BGR

    out = (base.astype(np.float32) * (1.0 - damp_mask_3c * 0.70) + damp_color * (damp_mask_3c * 0.70)).astype(np.uint8)

    # Add efflorescence / salt watermark fringe around drying boundary
    boundary = cv2.Canny((damp_mask * 255).astype(np.uint8), 80, 160)
    boundary = cv2.dilate(boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out[boundary > 0] = np.clip(out[boundary > 0].astype(np.int32) + 35, 0, 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def generate_fresco_peeling_sample(width: int = 960, height: int = 720) -> np.ndarray:
    """Generate painted plaster wall with severe flaking paint & exposed raw underlayer."""
    np.random.seed(202)
    # 1. Base painted surface (aged pale cream-ochre fresco lime plaster)
    painted_wall = np.full((height, width, 3), [190, 215, 235], dtype=np.uint8)  # BGR
    # Add subtle aged paint texture
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    painted_wall = np.clip(painted_wall.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Exposed underlayer: rough red terracotta brick & coarse lime mortar
    underlayer = np.full((height, width, 3), [60, 75, 155], dtype=np.uint8)  # Terracotta BGR
    # Add rough brick mortar grid
    for y in range(0, height, 65):
        cv2.line(underlayer, (0, y), (width, y), (140, 145, 150), 4)
    for row, y in enumerate(range(0, height, 65)):
        offset = 0 if row % 2 == 0 else 70
        for x in range(offset, width, 140):
            cv2.line(underlayer, (x, y), (x, y + 65), (140, 145, 150), 3)

    # 3. Create jagged peeling/flaking mask
    peel_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw large irregular peeling patches
    def draw_irregular_patch(mask, cx, cy, radius, num_pts=16):
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        rads = radius * (1.0 + np.random.uniform(-0.35, 0.45, num_pts))
        pts = np.array([
            [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
            for a, r in zip(angles, rads)
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    draw_irregular_patch(peel_mask, 320, 260, 140)
    draw_irregular_patch(peel_mask, 410, 360, 110)
    draw_irregular_patch(peel_mask, 700, 480, 120)
    draw_irregular_patch(peel_mask, 650, 220, 80)
    draw_irregular_patch(peel_mask, 200, 520, 90)

    # Roughen edges with morphology
    peel_mask = cv2.dilate(peel_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # Composite: where peel_mask is 255, show underlayer; else show painted wall
    out = painted_wall.copy()
    out[peel_mask > 0] = underlayer[peel_mask > 0]

    # Curling paint rim / edge ridges (shadow on one side, light lip on other)
    edges = cv2.Canny(peel_mask, 100, 200)
    shadow_rim = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    out[shadow_rim > 0] = (out[shadow_rim > 0].astype(np.float32) * 0.45).astype(np.uint8)

    highlight_rim = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out[highlight_rim > 0] = np.clip(out[highlight_rim > 0].astype(np.int32) + 60, 0, 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def generate_sculpture_moss_sample(width: int = 960, height: int = 720) -> np.ndarray:
    """Generate sandstone bas-relief with green algae & bryophyte moss colonies."""
    np.random.seed(303)
    # 1. Warm sandstone relief base
    base = np.full((height, width, 3), [140, 175, 205], dtype=np.uint8)  # Sandstone BGR
    # Add stone granular noise
    grain = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # 2. Carved relief contours (e.g. ancient dancing apsara / deity silhouette & halo)
    cv2.circle(base, (width // 2, 260), 140, (110, 140, 170), 12)  # Halo
    # Head & torso
    cv2.circle(base, (width // 2, 260), 65, (100, 130, 160), -1)
    cv2.ellipse(base, (width // 2, 450), (90, 140), 0, 0, 360, (100, 130, 160), -1)
    # Pillars on sides
    cv2.rectangle(base, (60, 40), (160, 680), (110, 140, 170), 8)
    cv2.rectangle(base, (width - 160, 40), (width - 60, 680), (110, 140, 170), 8)

    # 3. Biological colonization: moss & algae clustering in crevices and damp corners
    bio_mask = np.zeros((height, width), dtype=np.float32)

    # Clustered colonies around pillar bases and carving crevices
    colony_spots = [
        (110, 550, 60, 90), (140, 420, 45, 60), (width - 110, 520, 65, 100),
        (480, 540, 90, 70), (430, 360, 50, 45), (530, 370, 45, 50),
        (260, 620, 80, 50), (700, 630, 85, 55)
    ]
    for cx, cy, rx, ry in colony_spots:
        cv2.ellipse(bio_mask, (cx, cy), (rx, ry), np.random.randint(-20, 20), 0, 360, 1.0, -1)

    # Add high-frequency moss clump texture
    clump_noise = np.random.normal(0, 0.35, (height, width)).astype(np.float32)
    bio_mask = np.clip(cv2.GaussianBlur(bio_mask, (15, 15), 0) + clump_noise * 0.4, 0.0, 1.0)
    bio_mask = cv2.GaussianBlur(bio_mask, (5, 5), 0)

    # Deep green moss color with yellow-olive highlights
    moss_color_bgr = np.array([25, 135, 45], dtype=np.float32)   # Vibrant Moss Green BGR
    olive_color_bgr = np.array([20, 160, 110], dtype=np.float32) # Olive Lichen BGR

    out = base.copy().astype(np.float32)
    b_mask_3c = np.repeat(bio_mask[:, :, np.newaxis], 3, axis=2)

    # Blend moss into stone
    moss_layer = moss_color_bgr * 0.7 + olive_color_bgr * 0.3
    out = out * (1.0 - b_mask_3c * 0.85) + moss_layer * (b_mask_3c * 0.85)

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def generate_sandstone_erosion_sample(width: int = 960, height: int = 720) -> np.ndarray:
    """Generate weathered sandstone monument with granular pitting & surface erosion."""
    np.random.seed(404)
    # 1. Warm sandstone base
    base = np.full((height, width, 3), [135, 165, 210], dtype=np.uint8)  # BGR

    # 2. Add an ancient relief rosette / chakra in center
    cv2.circle(base, (width // 2, height // 2), 160, (105, 130, 175), 10)
    for r in [60, 110]:
        cv2.circle(base, (width // 2, height // 2), r, (105, 130, 175), 4)

    # 3. Severe localized erosion: granular pitting and loss of carved details
    # Create erosion roughness map
    erosion_zone = np.zeros((height, width), dtype=np.float32)
    for cx, cy, rx, ry in [
        (420, 320, 140, 120), (560, 400, 160, 130), (300, 480, 110, 90), (660, 260, 100, 80)
    ]:
        cv2.ellipse(erosion_zone, (cx, cy), (rx, ry), 15, 0, 360, 1.0, -1)

    erosion_zone = cv2.GaussianBlur(erosion_zone, (35, 35), 0)

    # Generate intense abrasive granular pits
    pit_noise = np.random.normal(0, 45, (height, width, 3)).astype(np.float32)
    # Alveolar honeycomb voids (salt weathering pockets)
    pockets = np.random.uniform(0, 1, (height, width)) > 0.96
    pit_noise[pockets] = -85.0

    out = base.astype(np.float32)
    zone_3c = np.repeat(erosion_zone[:, :, np.newaxis], 3, axis=2)

    # Weathered zones have high granular roughness and washed-out contrast
    out = out + pit_noise * zone_3c * 0.85
    # Slightly yellow/ochre bleaching from weathering
    out[:, :, 0] -= erosion_zone * 25.0

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def main():
    os.makedirs("sample_images", exist_ok=True)
    samples = [
        ("sample_images/sample1_temple_crack.jpg", generate_granite_crack_sample()),
        ("sample_images/sample2_mural_moisture.jpg", generate_mural_moisture_sample()),
        ("sample_images/sample3_fresco_peeling.jpg", generate_fresco_peeling_sample()),
        ("sample_images/sample4_sculpture_moss.jpg", generate_sculpture_moss_sample()),
        ("sample_images/sample5_sandstone_erosion.jpg", generate_sandstone_erosion_sample()),
    ]

    for path, img_rgb in samples:
        pil = Image.fromarray(img_rgb)
        pil.save(path, quality=92)
        print(f"Generated sample asset: {path} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")


if __name__ == "__main__":
    main()
