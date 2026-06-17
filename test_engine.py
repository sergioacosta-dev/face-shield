"""
test_engine.py — End-to-end test of the adversarial engine.

Downloads a Creative Commons / public-domain portrait from Wikimedia,
applies protection, then reports:
  • faces detected
  • cosine similarity before vs after (lower = more confused)
  • max & mean pixel difference (imperceptibility check)
  • PSNR (higher = more similar to original; >40 dB = imperceptible)
  • saves original.png, protected.png, diff_amplified.png
"""

import sys
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

# ---------------------------------------------------------------------------
# Download a public-domain test portrait (Wikimedia Commons, CC0)
# Small crop of a famous public-domain painting used for CV benchmarks.
# ---------------------------------------------------------------------------
TEST_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
)

# Use a real human face photo instead — public domain from Wikimedia
FACE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
)

# Better: use the classic "lena" style public domain test image that is
# commonly used in signal processing. We'll grab a small CC0 portrait.
PORTRAIT_URL = (
    "https://thispersondoesnotexist.com/"  # AI-generated, no real person
)

def download_test_image(path: str = "test_face.jpg") -> Image.Image:
    """Download a test portrait or create a synthetic one as fallback."""
    # Try to download a face from "This Person Does Not Exist"
    # (AI-generated faces — no real person, purely synthetic, no privacy issues)
    headers = {"User-Agent": "FaceShieldTest/1.0"}
    urls = [
        "https://thispersondoesnotexist.com/",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
            img = Image.open(path).convert("RGB")
            print(f"Downloaded test image from {url}  ({img.size[0]}×{img.size[1]})")
            return img
        except Exception as exc:
            print(f"  Download failed ({exc}), trying fallback…")

    # Fallback: create a synthetic 256×256 face-like image
    print("Using synthetic test image (gradient + texture).")
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    # Skin-tone background
    arr[:, :] = [210, 170, 140]
    # Darker oval face region
    cx, cy = 128, 128
    for y in range(256):
        for x in range(256):
            if ((x - cx) / 80) ** 2 + ((y - cy) / 100) ** 2 < 1:
                arr[y, x] = [190, 150, 120]
    # Eyes (dark)
    arr[100:115, 90:110]  = [40, 30, 25]
    arr[100:115, 146:166] = [40, 30, 25]
    # Nose
    arr[135:155, 120:136] = [170, 130, 100]
    # Mouth
    arr[168:178, 100:156] = [160, 80, 80]
    img = Image.fromarray(arr)
    img.save(path)
    return img


def compute_metrics(orig: Image.Image, prot: Image.Image) -> dict:
    a = np.array(orig, dtype=np.float64)
    b = np.array(prot, dtype=np.float64)
    diff = np.abs(a - b)
    mse  = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")
    return {
        "max_pixel_delta":  diff.max(),
        "mean_pixel_delta": diff.mean(),
        "psnr_db":          psnr,
    }


def embedding_similarity(model, t1: torch.Tensor, t2: torch.Tensor) -> float:
    with torch.no_grad():
        e1 = model(t1.unsqueeze(0))
        e2 = model(t2.unsqueeze(0))
        return F.cosine_similarity(e1, e2).item()


def save_diff_image(orig: Image.Image, prot: Image.Image, path: str, amplify: int = 15):
    """Save pixel difference amplified for visibility."""
    a = np.array(orig, dtype=np.int16)
    b = np.array(prot, dtype=np.int16)
    diff = np.clip(np.abs(a - b) * amplify, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(path)
    print(f"Difference image (×{amplify}) saved → {path}")


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    from adversarial_engine import AdversarialEngine

    print("=" * 60)
    print("Face Shield — Engine Test")
    print("=" * 60)

    # 1. Get test image
    print("\n[1] Loading test image…")
    orig = download_test_image("test_face.jpg")
    orig.save("original.png")
    print(f"    Saved → original.png  ({orig.size[0]}×{orig.size[1]})")

    # 2. Init engine
    print("\n[2] Initialising engine (downloads weights on first run)…")
    engine = AdversarialEngine()

    steps_done = []
    def progress(cur, tot):
        if cur == 1 or cur % 10 == 0 or cur == tot:
            pct = int(cur / tot * 100)
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r    [{bar}] {pct:3d}%  step {cur}/{tot}", end="", flush=True)
        steps_done.append(cur)

    engine.progress_callback = progress
    engine._load()
    print("\n    Models loaded.")

    # 3. Detect faces
    print("\n[3] Detecting faces…")
    faces = engine.detect_faces(orig)
    if faces:
        print(f"    {len(faces)} face(s) detected: {faces}")
    else:
        print("    No faces detected — will apply to full image.")

    # 4. Apply protection (quick test: 30 iterations)
    print("\n[4] Applying adversarial protection (30 iterations)…")
    steps_done.clear()
    protected = engine.protect_digital(
        orig, epsilon=0.05, iterations=30, face_only=True
    )
    print()   # newline after progress bar
    protected.save("protected.png")
    print(f"    Saved → protected.png")

    # 5. Pixel-level metrics
    print("\n[5] Imperceptibility metrics…")
    m = compute_metrics(orig, protected)
    print(f"    Max pixel Δ   : {m['max_pixel_delta']:.1f} / 255")
    print(f"    Mean pixel Δ  : {m['mean_pixel_delta']:.3f} / 255")
    print(f"    PSNR          : {m['psnr_db']:.1f} dB  (>40 dB = imperceptible)")

    # 6. Embedding similarity
    print("\n[6] Face embedding similarity…")
    t_orig = engine._to_tensor(orig.resize((160, 160)))
    t_prot = engine._to_tensor(protected.resize((160, 160)))
    sim = embedding_similarity(engine._model, t_orig, t_prot)
    print(f"    Cosine similarity: {sim:.4f}")
    print(f"    (1.0 = identical identity, 0.0 = completely different)")
    if sim < 0.5:
        print("    RESULT: Attack SUCCESSFUL — face identity confused.")
    elif sim < 0.8:
        print("    RESULT: Partial — increase iterations or epsilon for stronger effect.")
    else:
        print("    RESULT: Weak — try more iterations or a higher epsilon value.")

    # 7. Diff image
    print("\n[7] Saving amplified difference image…")
    save_diff_image(orig, protected, "diff_amplified.png", amplify=15)

    print("\n" + "=" * 60)
    print("Output files:")
    print("  original.png       — original image")
    print("  protected.png      — adversarially protected image")
    print("  diff_amplified.png — pixel differences ×15 (should look like noise)")
    print("=" * 60)


if __name__ == "__main__":
    main()
