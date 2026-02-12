"""Generate OG-letter favicon for OptionsGanster."""
from PIL import Image, ImageDraw, ImageFont
import os, sys

def generate_favicon():
    # Try to find a bold font
    bold_fonts = [
        "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
        "C:/Windows/Fonts/impact.ttf",     # Impact
        "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI
        "C:/Windows/Fonts/arial.ttf",      # Arial
    ]
    
    font_path = None
    for f in bold_fonts:
        if os.path.exists(f):
            font_path = f
            break

    for size in [16, 32, 48, 180]:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Dark rounded-square background
        pad = max(0, size // 16)
        r = size // 4
        draw.rounded_rectangle(
            [pad, pad, size - pad - 1, size - pad - 1],
            radius=r, fill=(10, 10, 26)
        )

        # "OG" text
        font_size = int(size * 0.52)
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        text = "OG"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Center text
        x = (size - tw) / 2 - bbox[0]
        y = (size - th) / 2 - bbox[1]

        # Draw text with red "O" and green "G"
        # First draw "O" in red
        o_bbox = draw.textbbox((0, 0), "O", font=font)
        o_w = o_bbox[2] - o_bbox[0]

        draw.text((x, y), "O", fill=(239, 83, 80), font=font)
        
        # Then draw "G" in green, offset by width of "O"
        # Get the exact position for G
        og_bbox = draw.textbbox((0, 0), "OG", font=font)
        g_only_bbox = draw.textbbox((0, 0), "G", font=font)
        
        # Use textlength for precise kerning
        try:
            o_advance = font.getlength("O")
        except AttributeError:
            o_advance = o_w + 1
        
        draw.text((x + o_advance, y), "G", fill=(76, 175, 80), font=font)

        out = os.path.join("app", "static", f"favicon-{size}.png")
        img.save(out, "PNG")
        print(f"  ok {out} ({os.path.getsize(out)} bytes)")

    # Build .ico with 16 + 32 + 48
    imgs = [Image.open(os.path.join("app", "static", f"favicon-{s}.png")) for s in [16, 32, 48]]
    ico_path = os.path.join("app", "static", "favicon.ico")
    imgs[1].save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)],
                 append_images=[imgs[0], imgs[2]])
    print(f"  ok {ico_path} ({os.path.getsize(ico_path)} bytes)")

    # Remove temp sizes
    for s in [16, 48]:
        p = os.path.join("app", "static", f"favicon-{s}.png")
        try:
            os.remove(p)
        except:
            pass

if __name__ == "__main__":
    generate_favicon()
