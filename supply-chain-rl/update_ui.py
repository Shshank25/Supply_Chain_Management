"""One-shot script to restyle the Gradio app with a purple/teal premium palette."""
import re

path = r"e:\Supply Chain Management\supply-chain-rl\app.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# ---------- 1. Google Font import (right after the opening triple-quote of APP_CSS) ----------
src = src.replace(
    'APP_CSS = """\n:root {',
    'APP_CSS = """\n'
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');\n"
    "\n:root {"
)

# ---------- 2. Design tokens ----------
token_map = {
    "--bg-top: #050d1a;":           "--bg-top: #08080f;",
    "--bg-mid: #081325;":           "--bg-mid: #0e0e1a;",
    "--bg-bottom: #101a31;":        "--bg-bottom: #151520;",
    "--panel: rgba(10, 18, 35, 0.78);":    "--panel: rgba(20, 20, 35, 0.72);",
    "--panel-strong: rgba(8, 16, 31, 0.92);":  "--panel-strong: rgba(14, 14, 26, 0.88);",
    "--panel-soft: rgba(17, 30, 56, 0.76);":   "--panel-soft: rgba(30, 30, 55, 0.6);",
    "--panel-border: rgba(110, 163, 255, 0.18);": "--panel-border: rgba(255, 255, 255, 0.07);",
    "--panel-glow: rgba(76, 215, 255, 0.18);":    "--panel-glow: rgba(168, 85, 247, 0.22);",
    "--ink: #edf5ff;":              "--ink: #f1f5f9;",
    "--muted: #a6b6d7;":           "--muted: #94a3b8;",
    "--accent: #57d8ff;":          "--accent: #a78bfa;",
    "--accent-2: #8dffb2;":        "--accent-2: #2dd4bf;",
    "--accent-3: #ffd27b;":        "--accent-3: #fbbf24;",
    "--accent-4: #7f9bff;":        "--accent-4: #f472b6;",
    "--warn: #ff8f7c;":            "--warn: #fb7185;",
    "--button-a: #2f73ff;":        "--button-a: #7c3aed;",
    "--button-b: #38d1ff;":        "--button-b: #a78bfa;",
}
for old, new in token_map.items():
    src = src.replace(old, new)

# ---------- 3. Font family ----------
src = src.replace(
    'font-family: "Bahnschrift", "Aptos", "Trebuchet MS", sans-serif;',
    "font-family: 'Inter', system-ui, -apple-system, sans-serif;",
)

# ---------- 4. Bulk RGB replacements for glow / shadow colours ----------
rgb_map = {
    "87, 216, 255":   "167, 139, 250",   # --accent  purple glow
    "141, 255, 178":  "45, 212, 191",     # --accent-2 teal glow
    "255, 210, 123":  "251, 191, 36",     # --accent-3 amber glow
    "127, 155, 255":  "244, 114, 182",    # --accent-4 pink glow
    "255, 143, 124":  "251, 113, 133",    # --warn glow
    "47, 115, 255":   "124, 58, 237",     # button shadow
    "56, 209, 255":   "167, 139, 250",    # button hover shadow
    "76, 215, 255":   "168, 85, 247",     # panel glow
}
for old_rgb, new_rgb in rgb_map.items():
    src = src.replace(old_rgb, new_rgb)

# ---------- 5. Background panel gradients ----------
src = src.replace(
    "linear-gradient(180deg, rgba(14, 24, 46, 0.92) 0%, rgba(10, 18, 34, 0.86) 100%)",
    "linear-gradient(180deg, rgba(18, 18, 32, 0.88) 0%, rgba(12, 12, 24, 0.82) 100%)",
)
src = src.replace(
    "linear-gradient(180deg, rgba(14, 28, 54, 0.78) 0%, rgba(9, 18, 35, 0.88) 100%)",
    "linear-gradient(180deg, rgba(22, 20, 38, 0.74) 0%, rgba(14, 12, 26, 0.84) 100%)",
)
src = src.replace(
    "linear-gradient(180deg, rgba(14, 24, 46, 0.9) 0%, rgba(9, 18, 35, 0.82) 100%)",
    "linear-gradient(180deg, rgba(20, 18, 34, 0.86) 0%, rgba(12, 10, 24, 0.78) 100%)",
)
src = src.replace(
    "linear-gradient(135deg, rgba(11, 20, 38, 0.94), rgba(20, 34, 64, 0.82))",
    "linear-gradient(135deg, rgba(16, 14, 30, 0.92), rgba(24, 20, 44, 0.8))",
)
src = src.replace(
    "linear-gradient(180deg, rgba(10, 20, 40, 0.94) 0%, rgba(8, 16, 31, 0.9) 100%)",
    "linear-gradient(180deg, rgba(18, 16, 34, 0.92) 0%, rgba(12, 10, 26, 0.88) 100%)",
)

# ---------- 6. Body-level background radials ----------
src = src.replace(
    "radial-gradient(circle at 12% 10%, rgba(167, 139, 250, 0.2), transparent 26%)",
    "radial-gradient(circle at 12% 10%, rgba(167, 139, 250, 0.18), transparent 30%)",
)

# ---------- 7. Enhance glassmorphism ----------
src = src.replace("backdrop-filter: blur(18px);", "backdrop-filter: blur(28px) saturate(160%);")

# ---------- 8. Chart colours (matplotlib) ----------
chart_map = {
    "trained_color = '#57d8ff'":       "trained_color = '#a78bfa'",
    "trained_fill = '#1588b8'":        "trained_fill  = '#7c3aed'",
    "random_color = '#ff8f7c'":        "random_color  = '#fb7185'",
    "random_fill = '#c65f49'":         "random_fill   = '#be123c'",
    "fulfillment_color = '#8dffb2'":   "fulfillment_color = '#2dd4bf'",
}
for old_c, new_c in chart_map.items():
    src = src.replace(old_c, new_c)

bg_chart_map = {
    "#081220": "#0c0c18",
    "#0b152b": "#0e0e1a",
    "#28405d": "#2a2a44",
    "#d9e8ff": "#e2e8f0",
    "#375474": "#3f3f66",
    "#091321": "#12121e",
    "#263f5c": "#2e2e4a",
    "#90a7c8": "#94a3b8",
    "#7ce6ff": "#c4b5fd",
    "#ffb2a5": "#fda4af",
}
for old_hex, new_hex in bg_chart_map.items():
    src = src.replace(old_hex, new_hex)

# ---------- 9. Signal bar gradients ----------
src = src.replace(
    "linear-gradient(90deg, #2e7aff 0%, #57d8ff 55%, #8dffb2 100%)",
    "linear-gradient(90deg, #7c3aed 0%, #a78bfa 55%, #2dd4bf 100%)",
)
src = src.replace(
    "linear-gradient(90deg, #ff8f7c 0%, #ffb38a 55%, #ffd27b 100%)",
    "linear-gradient(90deg, #fb7185 0%, #f9a8d4 55%, #fbbf24 100%)",
)
src = src.replace(
    "linear-gradient(90deg, #4d6fff 0%, #7f9bff 100%)",
    "linear-gradient(90deg, #7c3aed 0%, #a78bfa 100%)",
)

# ---------- 10. Scrollbar ----------
src = src.replace(
    "scrollbar-color: rgba(167, 139, 250, 0.42) rgba(7, 13, 28, 0.38);",
    "scrollbar-color: rgba(167, 139, 250, 0.5) rgba(12, 12, 20, 0.5);",
)
src = src.replace(
    "background: rgba(7, 13, 28, 0.42);",
    "background: rgba(12, 12, 20, 0.45);",
)
src = src.replace(
    "linear-gradient(180deg, rgba(167, 139, 250, 0.72), rgba(244, 114, 182, 0.52))",
    "linear-gradient(180deg, rgba(167, 139, 250, 0.7), rgba(45, 212, 191, 0.5))",
)
src = src.replace(
    "border: 2px solid rgba(7, 13, 28, 0.42);",
    "border: 2px solid rgba(12, 12, 20, 0.45);",
)

# ---------- 11. Input background ----------
src = src.replace(
    "background: rgba(7, 13, 28, 0.72) !important;",
    "background: rgba(14, 14, 24, 0.72) !important;",
)

# ---------- 12. Fill-between chart colour ----------
src = src.replace("color='#3d9f73'", "color='#0d9488'")

# ---------- Write ----------
with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("✅  UI restyled successfully – purple/teal palette, Inter font, enhanced glassmorphism.")
