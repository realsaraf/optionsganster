"""Generate OG image for OptionsGanster (1200x630)."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1200, 630
img = Image.new("RGB", (W, H), "#0a0a1a")
draw = ImageDraw.Draw(img)

# Background gradient-like effect with subtle circles
for cx, cy, r, color in [
    (600, 300, 400, (59, 130, 246, 12)),
    (300, 500, 250, (139, 92, 246, 8)),
    (900, 200, 300, (38, 166, 154, 8)),
]:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for ri in range(r, 0, -2):
        alpha = int(color[3] * (ri / r))
        od.ellipse(
            [cx - ri, cy - ri, cx + ri, cy + ri],
            fill=(color[0], color[1], color[2], alpha),
        )
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

draw = ImageDraw.Draw(img)

# Try system fonts
def get_font(size, bold=False):
    names = ["seguisb.ttf", "segoeui.ttf", "arial.ttf", "arialbd.ttf"] if not bold else ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    # Try Windows paths
    for name in names:
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()

font_big = get_font(64, bold=True)
font_med = get_font(28, bold=True)
font_sm = get_font(22)
font_badge = get_font(16, bold=True)

# Badge
badge_text = "AI-POWERED OPTIONS INTELLIGENCE"
bb = draw.textbbox((0, 0), badge_text, font=font_badge)
bw = bb[2] - bb[0]
bx = (W - bw) // 2
by = 130
# Badge background
draw.rounded_rectangle([bx - 16, by - 8, bx + bw + 16, by + 26], radius=14,
                        fill=(59, 130, 246, 30), outline=(59, 130, 246, 80))
draw.text((bx, by), badge_text, fill=(96, 165, 250), font=font_badge)

# Title line 1
t1 = "Stop Guessing."
bb1 = draw.textbbox((0, 0), t1, font=font_big)
draw.text(((W - (bb1[2] - bb1[0])) // 2, 190), t1, fill="#e0e0e0", font=font_big)

# Title line 2 (gradient-like - use blue/purple)
t2 = "Start Trading Smarter."
bb2 = draw.textbbox((0, 0), t2, font=font_big)
x2 = (W - (bb2[2] - bb2[0])) // 2
# Create gradient text
for i, ch in enumerate(t2):
    chbb = draw.textbbox((0, 0), t2[:i], font=font_big)
    cx = x2 + (chbb[2] - chbb[0])
    ratio = i / max(len(t2) - 1, 1)
    r = int(59 + (139 - 59) * ratio)
    g = int(130 + (92 - 130) * ratio)
    b = int(246 + (246 - 246) * ratio)
    draw.text((cx, 270), ch, fill=(r, g, b), font=font_big)

# Subtitle
sub = "Real-time AI signals. One actionable verdict."
bbs = draw.textbbox((0, 0), sub, font=font_sm)
draw.text(((W - (bbs[2] - bbs[0])) // 2, 365), sub, fill=(156, 163, 175), font=font_sm)

# Stats row
stats = [("AI", "Driven"), ("Live", "Signals"), ("10s", "Refresh"), ("$0", "Free")]
total_w = len(stats) * 140
sx = (W - total_w) // 2
for i, (val, label) in enumerate(stats):
    cx = sx + i * 140 + 70
    # Value
    vbb = draw.textbbox((0, 0), val, font=font_med)
    color = (96, 165, 250) if i % 2 == 0 else (38, 166, 154)
    draw.text((cx - (vbb[2] - vbb[0]) // 2, 430), val, fill=color, font=font_med)
    # Label
    lbb = draw.textbbox((0, 0), label, font=font_badge)
    draw.text((cx - (lbb[2] - lbb[0]) // 2, 470), label, fill=(107, 114, 128), font=font_badge)

# Thin bottom accent line
draw.rectangle([0, H - 4, W, H], fill=(59, 130, 246))

# Logo bottom left
logo_text = "OptionsGanster"
draw.text((40, H - 50), "Options", fill="#e0e0e0", font=font_badge)
obb = draw.textbbox((0, 0), "Options", font=font_badge)
draw.text((40 + obb[2] - obb[0], H - 50), "Ganster", fill=(239, 83, 80), font=font_badge)

# URL bottom right
url = "optionsganster.com"
ubb = draw.textbbox((0, 0), url, font=font_badge)
draw.text((W - 40 - (ubb[2] - ubb[0]), H - 50), url, fill=(107, 114, 128), font=font_badge)

out = os.path.join(os.path.dirname(__file__), "app", "static", "og-image.png")
img.save(out, "PNG", optimize=True)
print(f"Saved: {out} ({os.path.getsize(out)} bytes)")
