# Face Shield

Desktop GUI (Tkinter/CustomTkinter) that applies adversarial perturbations to
images and video to resist facial recognition and unauthorized AI model
training. Research/privacy tool, not a production service — no network
component, everything runs locally.

## What it does

Three tabs in `main.py`:

- **Digital Protection** — load a photo, apply imperceptible adversarial
  noise to detected face regions (or the whole image), save the result.
- **Printable Pattern** — generate a wearable adversarial pattern (glasses
  frame, patch, or full body panel) as a PNG or print-ready PDF.
- **Content Protection** — add a QR-code watermark and/or "training poison"
  noise to an image or video, to degrade any model trained on scraped copies.

## How the protection works (`adversarial_engine.py`)

- **Face evasion**: PGD (Projected Gradient Descent) attack against FaceNet
  (`facenet_pytorch`, InceptionResnetV1 pretrained on VGGFace2) as a surrogate
  model. Maximizes the angular distance between the original and perturbed
  face embeddings, within an ε-ball so the noise stays imperceptible.
  Adversarial examples are known to transfer across models to some degree, so
  this isn't limited to fooling FaceNet specifically — but it hasn't been
  validated here against a real production face-ID system, only measured via
  cosine distance on the FaceNet surrogate (see `test_engine.py` and the open
  item below).
- **Wearable patterns**: Adam-optimized patch that pushes its own FaceNet
  embedding as far as possible from any identity cluster. A separate body-patch
  mode targets Faster R-CNN (ResNet50-FPN v2) person detection instead, based
  on Thys et al. 2019 ("Fooling automated surveillance cameras").
  **Not empirically tested against a real camera or printed pattern** — only
  optimized/measured in tensor space.
- **Training poison**: error-maximizing PGD attack against a plain ResNet-50,
  based on Huang et al. 2021 ("Unlearnable Examples"). Intended to corrupt
  gradients if the image is scraped and used for training — not verified
  end-to-end against an actual training run.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install Pillow numpy customtkinter tqdm requests
pip install torch torchvision                 # CUDA build by default
pip install facenet-pytorch --no-deps
pip install qrcode opencv-python              # needed for the Content Protection tab
```

For CPU-only (smaller download), replace the torch line with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

FaceNet weights (~90 MB) download automatically on first use. Everything is
lazy-loaded — launching the app doesn't pull any models until you actually
apply a protection.

## Running

```bash
python main.py
```

## Testing

```bash
python test_engine.py
```

Downloads a public-domain portrait from Wikimedia, applies protection, and
reports faces detected, before/after cosine similarity, pixel-difference
stats, and PSNR (>40 dB = imperceptible to the eye). Saves
`original.png`/`protected.png`/`diff_amplified.png` for visual inspection.
This is the only test coverage that exists — there's no unit test suite for
the GUI or the individual engine methods.

## Known gaps

- No empirical validation against a real (non-surrogate) face-ID system —
  everything is measured via cosine distance on the FaceNet surrogate model.
- Wearable patterns (glasses/patch/body) are optimized and saved, but never
  tested against a live camera or an actual printout.
- Video output drops audio (`protect_video`); re-mux with ffmpeg if needed —
  see the note in `adversarial_engine.py`.
