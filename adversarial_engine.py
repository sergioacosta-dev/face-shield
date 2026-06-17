"""
adversarial_engine.py — Core engine for Face Shield.

Uses FaceNet (InceptionResnetV1 pretrained on VGGFace2) as a surrogate model.
PGD attack maximises cosine distance between the original and adversarial face
embeddings, creating imperceptible pixel-level noise that fools recognition
systems while being invisible to the human eye.

Typical epsilon values (perturbation in normalised [-1,1] space):
    0.02  →  ±2.5 pixel shift out of 255  (very subtle)
    0.05  →  ±6.4 pixel shift             (recommended default)
    0.10  →  ±12.7 pixel shift            (strong; borderline visible)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw


class AdversarialEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._detector = None
        # Set to a callable(current: int, total: int) to receive progress updates.
        self.progress_callback = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load FaceNet + MTCNN (downloads ~90 MB of weights on first call)."""
        if self._model is None:
            from facenet_pytorch import InceptionResnetV1
            self._model = (
                InceptionResnetV1(pretrained="vggface2")
                .eval()
                .to(self.device)
            )
        if self._detector is None:
            from facenet_pytorch import MTCNN
            self._detector = MTCNN(keep_all=True, device=self.device)

    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a FaceNet-normalised (3, 160, 160) tensor."""
        return T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])(pil_img.convert("RGB"))

    def _pgd(
        self,
        tensor: torch.Tensor,
        epsilon: float,
        iterations: int,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent — untargeted attack on FaceNet.

        Minimises cosine similarity between the original and adversarial
        embeddings (i.e. maximises angular distance → identity confusion).
        Returns the adversarial tensor in the same normalised [-1, 1] space.
        """
        alpha = epsilon / max(iterations // 5, 1)
        original = tensor.to(self.device)

        with torch.no_grad():
            orig_embed = self._model(original.unsqueeze(0))

        perturbed = original.clone()

        for i in range(iterations):
            perturbed = perturbed.detach().requires_grad_(True)
            adv_embed = self._model(perturbed.unsqueeze(0))

            # Minimise similarity → maximise distance from original identity
            loss = F.cosine_similarity(orig_embed, adv_embed).mean()
            loss.backward()

            with torch.no_grad():
                # Gradient descent step (negative direction = minimise loss)
                perturbed = perturbed - alpha * perturbed.grad.sign()
                # Project back into the epsilon-ball around the original
                delta = torch.clamp(perturbed - original, -epsilon, epsilon)
                perturbed = torch.clamp(original + delta, -1.0, 1.0)

            if self.progress_callback:
                self.progress_callback(i + 1, iterations)

        return perturbed.detach()

    def _extract_noise_pixels(
        self,
        orig_tensor: torch.Tensor,
        adv_tensor: torch.Tensor,
        target_wh: tuple,
    ) -> np.ndarray:
        """
        Convert the normalised noise to pixel-space and resize to target size.

        The FaceNet normalisation is:  t = pixel/127.5 - 1
        So the inverse is:            pixel = (t + 1) * 127.5
        And the pixel-space noise is: d_pixel = d_t * 127.5

        Returns a float32 array of shape (H, W, 3).
        """
        noise_norm = (adv_tensor.cpu() - orig_tensor).numpy()   # (3, 160, 160)
        noise_pix = noise_norm.transpose(1, 2, 0) * 127.5       # (160, 160, 3)

        # Shift to [0, 255] for clean BILINEAR resizing, then shift back
        noise_img = Image.fromarray(
            np.clip(noise_pix + 127.5, 0, 255).astype(np.uint8)
        ).resize(target_wh, Image.BILINEAR)

        return np.array(noise_img, dtype=np.float32) - 127.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: Image.Image) -> list:
        """
        Detect faces in the image.
        Returns a list of (x1, y1, x2, y2) bounding boxes (integers).
        """
        self._load()
        boxes, probs = self._detector.detect(image)
        if boxes is None:
            return []
        return [
            tuple(map(int, box))
            for box, p in zip(boxes, probs)
            if p > 0.85
        ]

    def protect_digital(
        self,
        image: Image.Image,
        epsilon: float = 0.05,
        iterations: int = 40,
        face_only: bool = True,
    ) -> Image.Image:
        """
        Add imperceptible adversarial noise that fools facial recognition.

        The noise is generated by PGD attack against FaceNet and then mapped
        back to pixel space.  Perturbations transfer to other recognition
        models (ArcFace, DeepFace, commercial APIs) via the transferability
        property of adversarial examples.

        Parameters
        ----------
        image      : Input PIL image.
        epsilon    : Perturbation magnitude (0.02–0.10 keeps it imperceptible).
        iterations : PGD steps. More = stronger protection, slower processing.
        face_only  : If True, perturb only detected face regions (faster).
                     Falls back to full image if no faces are found.
        """
        self._load()
        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb, dtype=np.float32)
        result = img_np.copy()

        if face_only:
            boxes = self.detect_faces(img_rgb)
            pad = 15
            regions = [
                (
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(img_np.shape[1], x2 + pad),
                    min(img_np.shape[0], y2 + pad),
                )
                for x1, y1, x2, y2 in boxes
            ]
            if not regions:
                regions = [None]   # no faces found → fall back to full image
        else:
            regions = [None]

        for region in regions:
            if region is not None:
                x1, y1, x2, y2 = region
                crop = Image.fromarray(img_np[y1:y2, x1:x2].astype(np.uint8))
            else:
                x1, y1 = 0, 0
                x2, y2 = img_np.shape[1], img_np.shape[0]
                crop = img_rgb

            orig_t = self._to_tensor(crop)
            adv_t = self._pgd(orig_t, epsilon, iterations)
            noise = self._extract_noise_pixels(orig_t, adv_t, (x2 - x1, y2 - y1))
            result[y1:y2, x1:x2] = np.clip(result[y1:y2, x1:x2] + noise, 0, 255)

        return Image.fromarray(result.astype(np.uint8))

    def generate_pattern(
        self,
        shape: str = "glasses",
        size: tuple = (600, 240),
        iterations: int = 120,
    ) -> Image.Image:
        """
        Generate a printable adversarial patch.

        The patch pixels are optimised (via Adam) to produce an embedding that
        lies as far as possible from any known-identity region of the embedding
        space.  When printed and worn (e.g. as glasses or a clothing patch),
        the patch pushes the wearer's face embedding out of the recognisable
        identity clusters.

        Parameters
        ----------
        shape      : 'glasses' for eyeglass-frame shaped patch, 'patch' for rectangle.
        size       : (width, height) in pixels.
        iterations : Optimisation steps.

        Returns an RGBA PIL image (transparent background for glasses shape).
        """
        self._load()
        W, H = size

        mask = (
            self._glasses_mask(W, H)
            if shape == "glasses"
            else torch.ones(1, 1, H, W)
        ).to(self.device)

        # Initialise patch with random values
        patch = (torch.rand(1, 3, H, W, device=self.device) * 2 - 1).requires_grad_(True)
        optimizer = torch.optim.Adam([patch], lr=0.008)

        for i in range(iterations):
            optimizer.zero_grad()
            masked = patch.clamp(-1, 1) * mask
            resized = F.interpolate(masked, (160, 160), mode="bilinear", align_corners=False)
            embed = self._model(resized)
            # Maximise the L2 norm of the embedding vector — push it away from
            # the origin of identity space so no identity is recognised.
            loss = -embed.norm(dim=1).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.data.clamp_(-1, 1)

            if self.progress_callback:
                self.progress_callback(i + 1, iterations)

        # Convert optimised patch to uint8 RGB
        p_np = patch.detach().cpu().clamp(-1, 1).squeeze(0)
        rgb = ((p_np.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

        # Build alpha channel from mask
        alpha = (mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
        if alpha.ndim == 3:
            alpha = alpha[0]

        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, "RGBA")

    def _load_body(self) -> None:
        """Lazy-load Faster R-CNN for person-detection evasion (~170 MB weights)."""
        if not hasattr(self, '_body_model') or self._body_model is None:
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._body_model = (
                fasterrcnn_resnet50_fpn_v2(weights=weights)
                .eval()
                .to(self.device)
            )

    def generate_body_pattern(
        self,
        size: tuple = (400, 400),
        iterations: int = 150,
    ) -> Image.Image:
        """
        Generate an adversarial body patch that suppresses person detection.

        Optimised against Faster R-CNN (ResNet50-FPN v2): the patch minimises
        the RPN objectness scores so the detector stops proposing the wearer's
        torso as a 'person' region.

        Based on: Thys et al. (2019) "Fooling automated surveillance cameras:
        adversarial patches to attack person detection."

        Print at ~30 × 30 cm and wear on your chest or back.
        Returns an RGB PIL image (no transparency needed — meant to be printed
        on a full surface like a T-shirt or vest panel).
        """
        self._load_body()
        W, H = size

        # Optimisable patch in [-1, 1] space
        patch = (torch.rand(1, 3, H, W, device=self.device) * 2 - 1).requires_grad_(True)
        optimizer = torch.optim.Adam([patch], lr=0.01)

        # Context canvas: simulates the patch sitting on a person-sized image.
        # Faster R-CNN expects [0, 1] float tensors at a reasonable resolution.
        ctx = 640

        for i in range(iterations):
            optimizer.zero_grad()

            # Normalise to [0, 1]
            patch_01 = (patch.clamp(-1, 1) + 1) / 2

            # Place patch in the centre 50% of the context (chest/torso region)
            ph = pw = int(ctx * 0.5)
            y0 = x0 = (ctx - ph) // 2

            # Build a differentiable context image:
            # patch pixels in the centre, neutral gray (0.5) outside
            mask = torch.zeros(1, 1, ctx, ctx, device=self.device)
            mask[:, :, y0:y0 + ph, x0:x0 + pw] = 1.0
            patch_full = F.interpolate(
                patch_01, (ctx, ctx), mode='bilinear', align_corners=False
            )
            context = patch_full * mask + 0.5 * (1 - mask)

            # FPN backbone → multi-scale feature maps
            features = self._body_model.backbone(context)

            # RPN head → objectness logits (one tensor per FPN level)
            objectness_logits, _ = self._body_model.rpn.head(list(features.values()))

            # Minimise objectness probability → suppress 'person' proposals
            loss = sum(torch.sigmoid(o).mean() for o in objectness_logits)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.data.clamp_(-1, 1)

            if self.progress_callback:
                self.progress_callback(i + 1, iterations)

        p_np = (
            (patch.detach().cpu().clamp(-1, 1)
             .squeeze(0).permute(1, 2, 0).numpy() + 1) * 127.5
        ).astype(np.uint8)
        return Image.fromarray(p_np, 'RGB')

    def _glasses_mask(self, W: int, H: int) -> torch.Tensor:
        """Return a glasses-frame shaped binary mask as a (1, 1, H, W) tensor."""
        img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(img)

        cx, cy = W // 2, H // 2
        lens_w = W // 4
        lens_h = int(H * 0.72)
        gap = max(W // 18, 6)
        bar_h = max(H // 6, 8)

        # Left lens ellipse
        lx1, lx2 = cx - gap - lens_w, cx - gap
        draw.ellipse([lx1, cy - lens_h // 2, lx2, cy + lens_h // 2], fill=220)

        # Right lens ellipse
        rx1, rx2 = cx + gap, cx + gap + lens_w
        draw.ellipse([rx1, cy - lens_h // 2, rx2, cy + lens_h // 2], fill=220)

        # Bridge
        draw.rectangle([cx - gap, cy - bar_h // 2, cx + gap, cy + bar_h // 2], fill=220)

        # Temples (arms extending to edges)
        draw.rectangle([0, cy - bar_h // 2, lx1, cy + bar_h // 2], fill=220)
        draw.rectangle([rx2, cy - bar_h // 2, W, cy + bar_h // 2], fill=220)

        mask_np = (
            np.array(img, dtype=np.float32)[np.newaxis, np.newaxis] / 220.0
        ).clip(0.0, 1.0)
        return torch.tensor(mask_np)

    # ------------------------------------------------------------------
    # Content protection — QR watermark + training poison + video
    # ------------------------------------------------------------------

    def _load_poison_model(self) -> None:
        """Lazy-load ResNet-50 for training-poison generation."""
        if not hasattr(self, '_poison_model') or self._poison_model is None:
            from torchvision.models import resnet50, ResNet50_Weights
            self._poison_model = (
                resnet50(weights=ResNet50_Weights.DEFAULT)
                .eval()
                .to(self.device)
            )

    def add_qr_watermark(
        self,
        image: Image.Image,
        text: str,
        opacity: float = 0.35,
        position: str = "bottom-right",
        size_frac: float = 0.18,
    ) -> Image.Image:
        """
        Overlay a semi-transparent QR code on the image.

        Parameters
        ----------
        text       : Content encoded in the QR (e.g. copyright notice / URL).
        opacity    : 0.0 = invisible, 1.0 = fully opaque.
        position   : 'bottom-right', 'bottom-left', 'top-right', 'top-left', 'center'.
        size_frac  : QR code width as a fraction of image width.
        """
        import qrcode as _qrcode

        img_rgb = image.convert("RGB")
        W, H = img_rgb.size

        # Generate QR
        qr = _qrcode.QRCode(
            version=None,
            error_correction=_qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=2,
        )
        qr.add_data(text)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

        # Resize QR to size_frac of image width
        qr_w = max(40, int(W * size_frac))
        qr_img = qr_img.resize((qr_w, qr_w), Image.NEAREST)

        # Apply opacity
        r, g, b, a = qr_img.split()
        a = a.point(lambda x: int(x * opacity))
        qr_img.putalpha(a)

        # Position
        pad = 10
        pos_map = {
            "bottom-right": (W - qr_w - pad, H - qr_w - pad),
            "bottom-left":  (pad, H - qr_w - pad),
            "top-right":    (W - qr_w - pad, pad),
            "top-left":     (pad, pad),
            "center":       ((W - qr_w) // 2, (H - qr_w) // 2),
        }
        x, y = pos_map.get(position, pos_map["bottom-right"])

        canvas = img_rgb.convert("RGBA")
        canvas.paste(qr_img, (x, y), qr_img)
        return canvas.convert("RGB")

    def add_training_poison(
        self,
        image: Image.Image,
        epsilon: float = 0.04,
        iterations: int = 40,
    ) -> Image.Image:
        """
        Add imperceptible error-maximising noise that corrupts AI training.

        If this image is scraped and used to train or fine-tune a model, the
        adversarial perturbation pushes the model's weights in the wrong
        direction — degrading what it learns from this image.

        Based on: Huang et al. (2021) "Unlearnable Examples: Making Personal
        Data Unexploitable." ICLR 2021.

        epsilon : perturbation in [0,1] space.
                  0.03 ≈ ±7.6 px  (subtle)
                  0.05 ≈ ±12.7 px (stronger)
        """
        self._load_poison_model()

        # ResNet expects [0,1] tensors at 224×224
        to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        img_rgb = image.convert("RGB")
        orig_size = img_rgb.size

        t = to_tensor(img_rgb).to(self.device)
        original = t.clone()

        # Detect dominant ImageNet class to anchor the attack
        with torch.no_grad():
            logits = self._poison_model(original.unsqueeze(0))
            label = logits.argmax(dim=1)

        alpha = epsilon / max(iterations // 5, 1)
        perturbed = original.clone()

        for i in range(iterations):
            perturbed = perturbed.detach().requires_grad_(True)
            logits = self._poison_model(perturbed.unsqueeze(0))

            # MAXIMISE cross-entropy → error-maximising perturbation
            # Any model trained on this image will have its gradients corrupted
            loss = -F.cross_entropy(logits, label)
            loss.backward()

            with torch.no_grad():
                perturbed = perturbed - alpha * perturbed.grad.sign()
                delta = torch.clamp(perturbed - original, -epsilon, epsilon)
                perturbed = torch.clamp(original + delta, 0.0, 1.0)

            if self.progress_callback:
                self.progress_callback(i + 1, iterations)

        # Extract pixel-space noise and apply to original resolution
        noise = (perturbed - original).cpu().numpy()           # (3, 224, 224)
        noise_pil = Image.fromarray(
            np.clip(noise.transpose(1, 2, 0) * 127.5 + 127.5, 0, 255).astype(np.uint8)
        ).resize(orig_size, Image.BILINEAR)
        noise_np = np.array(noise_pil, dtype=np.float32) - 127.5  # back to pixel deltas

        img_np = np.array(img_rgb, dtype=np.float32)
        poisoned = np.clip(img_np + noise_np, 0, 255).astype(np.uint8)
        return Image.fromarray(poisoned)

    def protect_video(
        self,
        input_path: str,
        output_path: str,
        qr_text: str | None = None,
        qr_opacity: float = 0.35,
        qr_position: str = "bottom-right",
        add_poison: bool = True,
        poison_epsilon: float = 0.04,
        poison_iterations: int = 20,
        frame_callback=None,   # callable(current_frame, total_frames)
    ) -> None:
        """
        Apply QR watermark and/or training poison to every frame of a video.

        Output is written as an MP4 (audio is not preserved — re-add with
        a tool like ffmpeg: ffmpeg -i original.mp4 -i protected.mp4 -c copy out.mp4).

        Parameters
        ----------
        input_path  : Path to the source video file.
        output_path : Path for the protected output video.
        qr_text     : If set, overlay a QR code watermark on every frame.
        add_poison  : If True, apply training-poison perturbation to each frame.
        frame_callback : callable(current, total) for progress reporting.
        """
        import cv2

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Pre-load poison model once outside the loop
        if add_poison:
            self._load_poison_model()

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # OpenCV uses BGR; convert to PIL RGB
            pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            if add_poison:
                # Suppress per-step callback inside per-frame loop
                saved_cb = self.progress_callback
                self.progress_callback = None
                pil_frame = self.add_training_poison(
                    pil_frame, epsilon=poison_epsilon, iterations=poison_iterations
                )
                self.progress_callback = saved_cb

            if qr_text:
                pil_frame = self.add_qr_watermark(
                    pil_frame, qr_text,
                    opacity=qr_opacity,
                    position=qr_position,
                )

            # Convert back to BGR for OpenCV
            out_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            out.write(out_frame)

            frame_idx += 1
            if frame_callback:
                frame_callback(frame_idx, total)

        cap.release()
        out.release()
