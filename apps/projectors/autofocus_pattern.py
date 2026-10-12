import math
from PIL import Image, ImageDraw, ImageFont

# Generate a projector auto-focus calibration pattern (1920x1080).
# Combines Siemens star, checkerboard elements, and resolution targets.
def generate_pattern(output):
    W, H = 1920, 1080
    img = Image.new("L", (W, H), 128)  # gray background
    draw = ImageDraw.Draw(img)

    # ── 1. Checkerboard corners ──────────────────────────────────────
    check_size = 40
    corners = [
        (0, 0, W // 3, H // 3),
        (W - W // 3, 0, W, H // 3),
        (0, H - H // 3, W // 3, H),
        (W - W // 3, H - H // 3, W, H),
    ]
    for cx0, cy0, cx1, cy1 in corners:
        for y in range(cy0, cy1, check_size):
            for x in range(cx0, cx1, check_size):
                if ((x // check_size) + (y // check_size)) % 2 == 0:
                    draw.rectangle([x, y, x + check_size - 1, y + check_size - 1], fill=0)

    # ── 2. Siemens star (center) ─────────────────────────────────────
    cx, cy = W // 2, H // 2
    outer_r = 380
    inner_r = 8
    n_sectors = 72  # 36 white + 36 black

    for i in range(n_sectors):
        angle_start = 2 * math.pi * i / n_sectors
        angle_end = 2 * math.pi * (i + 1) / n_sectors
        fill = 255 if i % 2 == 0 else 0
        # draw filled sector polygon
        steps = 60  # smooth arc approximation
        points_outer = []
        points_inner = []
        for j in range(steps + 1):
            a = angle_start + (angle_end - angle_start) * j / steps
            points_outer.append((cx + outer_r * math.cos(a), cy + outer_r * math.sin(a)))
        for j in range(steps, -1, -1):
            a = angle_start + (angle_end - angle_start) * j / steps
            points_inner.append((cx + inner_r * math.cos(a), cy + inner_r * math.sin(a)))
        draw.polygon(points_outer + points_inner, fill=fill)

    # white ring around star
    draw.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r], outline=255, width=2)
    # black ring
    draw.ellipse([cx - outer_r - 10, cy - outer_r - 10, cx + outer_r + 10, cy + outer_r + 10], outline=0, width=2)

    # ── 3. Vertical / horizontal resolution bars ─────────────────────
    bar_h = 15
    bar_gap = 4
    freqs = [1, 2, 4, 8, 16, 32]  # pixel periods
    label_sizes = [f"1px line pairs ({f}lp/mm)" for f in [0.5, 1, 2, 4, 8, 16]]

    # horizontal bars (vertical resolution)
    x0, y0 = W // 2 - 350, H - 120
    for fi, f in enumerate(freqs):
        for col in range(0, 300, f * 2):
            draw.rectangle([x0 + col, y0 + fi * (bar_h + bar_gap), x0 + col + f - 1, y0 + fi * (bar_h + bar_gap) + bar_h - 1], fill=0)

    # vertical bars (horizontal resolution)
    x0 = W // 2 + 50
    for fi, f in enumerate(freqs):
        for row in range(0, 300, f * 2):
            draw.rectangle([x0 + fi * (bar_h + bar_gap), y0 + row, x0 + fi * (bar_h + bar_gap) + bar_h - 1, y0 + row + f - 1], fill=0)

    # ── 4. Crosshair ─────────────────────────────────────────────────
    draw.line([cx - 20, cy, cx + 20, cy], fill=255, width=2)
    draw.line([cx, cy - 20, cx, cy + 20], fill=255, width=2)
    draw.line([cx - 10, cy, cx + 10, cy], fill=0, width=1)
    draw.line([cx, cy - 10, cx, cy + 10], fill=0, width=1)

    # ── 5. Resolution circles ────────────────────────────────────────
    for r in [40, 80, 120, 160, 200, 240, 280, 320, 360]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=64, width=1)

    # ── 6. Edge targets (high-contrast rectangles around star) ───────
    placements = [
        (W // 4 - 80, H // 4 - 60),
        (W * 3 // 4 - 80, H // 4 - 60),
        (W // 4 - 80, H * 3 // 4 - 60),
        (W * 3 // 4 - 80, H * 3 // 4 - 60),
    ]
    for px, py in placements:
        draw.rectangle([px, py, px + 160, py + 120], fill=0, outline=255, width=3)
        # inner white square
        draw.rectangle([px + 40, py + 30, px + 120, py + 90], fill=255)
        # diagonal lines
        draw.line([px, py, px + 160, py + 120], fill=255, width=2)
        draw.line([px + 160, py, px, py + 120], fill=255, width=2)
        draw.line([px + 40, py + 30, px + 120, py + 90], fill=0, width=1)
        draw.line([px + 120, py + 30, px + 40, py + 90], fill=0, width=1)

    # ── 7. Text label ─────────────────────────────────────────────────
    try:
        font_large = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((W // 2 - 240, 20), "PROJECTOR AUTO-FOCUS TARGET", fill=0, font=font_large)
    draw.text((W // 2 - 200, 58), "Siemens Star + Checkerboard + Resolution Bars", fill=64, font=font_small)
    draw.text((W // 2 - 42, H - 30), "1920 x 1080", fill=192, font=font_small)

    img.save(output, "PNG")
    print(f"Saved: {output} ({img.size[0]}x{img.size[1]})")

if __name__ == "__main__":
    output = r"D:/ws/images/autofocus_target.png"
    generate_pattern(output)