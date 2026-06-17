"""
main.py — Face Shield GUI

Two-tab application:
  • Digital Protection  — load a photo, apply imperceptible adversarial noise,
                          save the protected image.
  • Printable Pattern   — generate an adversarial glasses / patch pattern,
                          save as PNG (with transparency) or print-ready PDF.
"""

from __future__ import annotations

import threading
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

QUALITY_ITERS: dict[str, int] = {"Fast": 20, "Medium": 40, "Best": 80}

PATTERN_SIZES: dict[str, dict[str, tuple]] = {
    "glasses": {"Small": (400, 160), "Medium": (600, 240), "Large": (900, 360)},
    "patch":   {"Small": (200, 200), "Medium": (300, 300), "Large": (500, 500)},
    "body":    {"Small": (300, 300), "Medium": (400, 400), "Large": (500, 500)},
}

# Physical print size for body patches (cm wide) — scales PDF output
BODY_PATCH_PRINT_CM = 30


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

class FaceShieldApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Shield — AI Privacy Protection")
        self.geometry("1120x730")
        self.minsize(900, 600)

        self._engine = None
        self._processing = False

        self.original_image: Image.Image | None = None
        self.result_image:   Image.Image | None = None
        self.pattern_image:  Image.Image | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Engine (lazy-loaded)
    # ------------------------------------------------------------------

    def _get_engine(self):
        if self._engine is None:
            from adversarial_engine import AdversarialEngine
            self._engine = AdversarialEngine()
        return self._engine

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Header ──────────────────────────────────────────────────
        header = ctk.CTkFrame(self, corner_radius=0, height=54)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="Face Shield",
            font=ctk.CTkFont(size=21, weight="bold"),
        ).pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            header,
            text="AI-Evasion Image Protection  ·  Privacy & Research Tool",
            text_color="gray",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", pady=10)

        # ── Tabs ────────────────────────────────────────────────────
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.tabs.add("Digital Protection")
        self.tabs.add("Printable Pattern")
        self.tabs.add("Content Protection")

        self._build_digital_tab(self.tabs.tab("Digital Protection"))
        self._build_pattern_tab(self.tabs.tab("Printable Pattern"))
        self._build_content_tab(self.tabs.tab("Content Protection"))

    # ── Digital tab ─────────────────────────────────────────────────

    def _build_digital_tab(self, parent: ctk.CTkFrame) -> None:
        # Image panels
        panels = ctk.CTkFrame(parent, fg_color="transparent")
        panels.pack(side="left", fill="both", expand=True)

        # Original
        orig_card = ctk.CTkFrame(panels)
        orig_card.pack(side="left", fill="both", expand=True, padx=(0, 4), pady=4)
        ctk.CTkLabel(orig_card, text="Original", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))
        self.orig_lbl = ctk.CTkLabel(
            orig_card,
            text="Click here to load an image\n\nPNG · JPG · BMP · WebP",
            width=390, height=420,
            fg_color=("#dde0e8", "#1a1a2e"),
            text_color="gray",
            cursor="hand2",
        )
        self.orig_lbl.pack(padx=8, pady=(0, 8), fill="both", expand=True)
        self.orig_lbl.bind("<Button-1>", lambda _: self._browse_image())

        # Protected result
        result_card = ctk.CTkFrame(panels)
        result_card.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        ctk.CTkLabel(result_card, text="Protected", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))
        self.result_lbl = ctk.CTkLabel(
            result_card,
            text="Protected image will appear here\nafter applying protection",
            width=390, height=420,
            fg_color=("#dde0e8", "#1a1a2e"),
            text_color="gray",
        )
        self.result_lbl.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        # Controls sidebar
        ctrl = ctk.CTkFrame(parent, width=235)
        ctrl.pack(side="right", fill="y", padx=(12, 0), pady=4)
        ctrl.pack_propagate(False)

        ctk.CTkLabel(ctrl, text="Settings", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(16, 8))

        # Strength slider
        ctk.CTkLabel(ctrl, text="Protection Strength", anchor="w").pack(fill="x", padx=16)
        self.d_eps = ctk.DoubleVar(value=0.05)
        self.d_eps_lbl = ctk.CTkLabel(ctrl, text="Medium  (ε = 0.050)", text_color="gray",
                                      font=ctk.CTkFont(size=11))
        ctk.CTkSlider(
            ctrl, from_=0.01, to=0.12, variable=self.d_eps,
            command=self._on_eps_change,
        ).pack(fill="x", padx=16, pady=(3, 0))
        self.d_eps_lbl.pack(pady=(0, 4))

        # Quality
        ctk.CTkLabel(ctrl, text="Quality", anchor="w").pack(fill="x", padx=16, pady=(10, 2))
        self.d_quality = ctk.StringVar(value="Medium")
        ctk.CTkSegmentedButton(
            ctrl, values=list(QUALITY_ITERS), variable=self.d_quality,
        ).pack(fill="x", padx=16)

        # Scope
        ctk.CTkLabel(ctrl, text="Apply To", anchor="w").pack(fill="x", padx=16, pady=(10, 2))
        self.d_scope = ctk.StringVar(value="Faces Only")
        ctk.CTkSegmentedButton(
            ctrl, values=["Faces Only", "Full Image"], variable=self.d_scope,
        ).pack(fill="x", padx=16)

        # Progress
        ctk.CTkFrame(ctrl, height=14, fg_color="transparent").pack()
        self.d_progress = ctk.CTkProgressBar(ctrl)
        self.d_progress.pack(fill="x", padx=16, pady=(0, 2))
        self.d_progress.set(0)
        self.d_status = ctk.CTkLabel(ctrl, text="Ready", text_color="gray",
                                     font=ctk.CTkFont(size=11))
        self.d_status.pack()

        # Buttons
        ctk.CTkFrame(ctrl, height=12, fg_color="transparent").pack()
        ctk.CTkButton(
            ctrl, text="Apply Protection",
            fg_color="#c0392b", hover_color="#922b21",
            font=ctk.CTkFont(weight="bold"),
            command=self._apply_protection,
        ).pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(
            ctrl, text="Save Result",
            command=self._save_digital,
        ).pack(fill="x", padx=16)

    # ── Pattern tab ─────────────────────────────────────────────────

    def _build_pattern_tab(self, parent: ctk.CTkFrame) -> None:
        # Pattern display
        left = ctk.CTkFrame(parent)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4), pady=4)
        ctk.CTkLabel(left, text="Generated Pattern", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))
        self.pattern_lbl = ctk.CTkLabel(
            left,
            text="Pattern will appear here after generation",
            fg_color=("#dde0e8", "#1a1a2e"),
            text_color="gray",
        )
        self.pattern_lbl.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        # Controls sidebar
        ctrl = ctk.CTkFrame(parent, width=255)
        ctrl.pack(side="right", fill="y", padx=(12, 0), pady=4)
        ctrl.pack_propagate(False)

        ctk.CTkLabel(ctrl, text="Pattern Settings",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(16, 8))

        ctk.CTkLabel(
            ctrl,
            text=(
                "Generates a wearable adversarial pattern optimised to confuse "
                "facial recognition cameras.\n\n"
                "Print it and wear as glasses frames or a clothing patch — "
                "the pattern pushes your face embedding away from any "
                "recognisable identity."
            ),
            text_color="gray",
            wraplength=225,
            justify="left",
            font=ctk.CTkFont(size=11),
        ).pack(padx=12, pady=(0, 14))

        # Shape
        ctk.CTkLabel(ctrl, text="Pattern Shape", anchor="w").pack(fill="x", padx=16)
        self.p_shape = ctk.StringVar(value="Glasses")
        ctk.CTkSegmentedButton(
            ctrl, values=["Glasses", "Full Patch", "Body Patch"],
            variable=self.p_shape,
            command=self._on_shape_change,
        ).pack(fill="x", padx=16)

        # Body patch hint (shown/hidden depending on selection)
        self.body_hint = ctk.CTkLabel(
            ctrl,
            text="Print at 30×30 cm, wear on chest or back.\n"
                 "Optimised against Faster R-CNN person detector.",
            text_color="#e8a020",
            wraplength=220,
            justify="left",
            font=ctk.CTkFont(size=11),
        )
        # hidden by default

        # Size
        ctk.CTkLabel(ctrl, text="Output Size", anchor="w").pack(fill="x", padx=16, pady=(10, 2))
        self.p_size = ctk.StringVar(value="Medium")
        ctk.CTkSegmentedButton(
            ctrl, values=["Small", "Medium", "Large"], variable=self.p_size,
        ).pack(fill="x", padx=16)

        # Quality
        ctk.CTkLabel(ctrl, text="Quality (more = stronger)", anchor="w").pack(
            fill="x", padx=16, pady=(10, 2))
        self.p_quality = ctk.StringVar(value="Medium")
        ctk.CTkSegmentedButton(
            ctrl, values=list(QUALITY_ITERS), variable=self.p_quality,
        ).pack(fill="x", padx=16)

        # Progress
        ctk.CTkFrame(ctrl, height=14, fg_color="transparent").pack()
        self.p_progress = ctk.CTkProgressBar(ctrl)
        self.p_progress.pack(fill="x", padx=16, pady=(0, 2))
        self.p_progress.set(0)
        self.p_status = ctk.CTkLabel(ctrl, text="Ready", text_color="gray",
                                     font=ctk.CTkFont(size=11))
        self.p_status.pack()

        # Buttons
        ctk.CTkFrame(ctrl, height=12, fg_color="transparent").pack()
        ctk.CTkButton(
            ctrl, text="Generate Pattern",
            fg_color="#c0392b", hover_color="#922b21",
            font=ctk.CTkFont(weight="bold"),
            command=self._generate_pattern,
        ).pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(
            ctrl, text="Save as PNG (with transparency)",
            command=self._save_pattern_png,
        ).pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkButton(
            ctrl, text="Save as PDF (print-ready)",
            command=self._save_pattern_pdf,
        ).pack(fill="x", padx=16)

    # ── Content Protection tab ──────────────────────────────────────

    def _build_content_tab(self, parent: ctk.CTkFrame) -> None:
        # Preview panel (left)
        left = ctk.CTkFrame(parent)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4), pady=4)
        ctk.CTkLabel(left, text="Preview", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))
        self.cp_preview = ctk.CTkLabel(
            left,
            text="Load an image or video to preview it here",
            fg_color=("#dde0e8", "#1a1a2e"),
            text_color="gray",
        )
        self.cp_preview.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        self.cp_file_lbl = ctk.CTkLabel(left, text="No file loaded", text_color="gray",
                                         font=ctk.CTkFont(size=11))
        self.cp_file_lbl.pack(pady=(0, 6))

        # Controls (right)
        ctrl = ctk.CTkFrame(parent, width=280)
        ctrl.pack(side="right", fill="y", padx=(12, 0), pady=4)
        ctrl.pack_propagate(False)

        ctk.CTkLabel(ctrl, text="Content Protection",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(16, 4))
        ctk.CTkLabel(
            ctrl,
            text="Protects images & videos from being scraped\nand used to train AI models.",
            text_color="gray", font=ctk.CTkFont(size=11), wraplength=250,
        ).pack(padx=12, pady=(0, 12))

        # Load buttons
        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkButton(btn_row, text="Load Image", width=120,
                      command=self._cp_load_image).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_row, text="Load Video", width=120,
                      command=self._cp_load_video).pack(side="left")

        ctk.CTkFrame(ctrl, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=8)

        # QR watermark section
        ctk.CTkLabel(ctrl, text="QR Watermark", font=ctk.CTkFont(weight="bold"),
                     anchor="w").pack(fill="x", padx=12)
        self.cp_qr_enable = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl, text="Add QR watermark", variable=self.cp_qr_enable).pack(
            anchor="w", padx=12, pady=(4, 0))

        ctk.CTkLabel(ctrl, text="Watermark text / URL", anchor="w",
                     font=ctk.CTkFont(size=11)).pack(fill="x", padx=12, pady=(6, 0))
        self.cp_qr_text = ctk.CTkEntry(ctrl, placeholder_text="© Your Name — No AI Training")
        self.cp_qr_text.pack(fill="x", padx=12, pady=(2, 0))

        ctk.CTkLabel(ctrl, text="Opacity", anchor="w",
                     font=ctk.CTkFont(size=11)).pack(fill="x", padx=12, pady=(6, 0))
        self.cp_opacity = ctk.DoubleVar(value=0.35)
        self.cp_opacity_lbl = ctk.CTkLabel(ctrl, text="35%", text_color="gray",
                                            font=ctk.CTkFont(size=11))
        ctk.CTkSlider(ctrl, from_=0.1, to=1.0, variable=self.cp_opacity,
                      command=lambda v: self.cp_opacity_lbl.configure(
                          text=f"{int(float(v)*100)}%")
                      ).pack(fill="x", padx=12)
        self.cp_opacity_lbl.pack()

        ctk.CTkLabel(ctrl, text="Position", anchor="w",
                     font=ctk.CTkFont(size=11)).pack(fill="x", padx=12, pady=(4, 0))
        self.cp_qr_pos = ctk.StringVar(value="bottom-right")
        ctk.CTkOptionMenu(ctrl, values=["bottom-right", "bottom-left",
                                         "top-right", "top-left", "center"],
                          variable=self.cp_qr_pos).pack(fill="x", padx=12)

        ctk.CTkFrame(ctrl, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=8)

        # Training poison section
        ctk.CTkLabel(ctrl, text="AI Training Poison",
                     font=ctk.CTkFont(weight="bold"), anchor="w").pack(fill="x", padx=12)
        ctk.CTkLabel(
            ctrl,
            text="Adds invisible noise that corrupts model training\nif this image is scraped.",
            text_color="gray", font=ctk.CTkFont(size=11), wraplength=250, justify="left",
        ).pack(padx=12, pady=(2, 4))
        self.cp_poison_enable = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(ctrl, text="Enable training poison",
                        variable=self.cp_poison_enable).pack(anchor="w", padx=12)

        ctk.CTkLabel(ctrl, text="Poison strength", anchor="w",
                     font=ctk.CTkFont(size=11)).pack(fill="x", padx=12, pady=(6, 0))
        self.cp_poison_str = ctk.StringVar(value="Medium")
        ctk.CTkSegmentedButton(ctrl, values=["Subtle", "Medium", "Strong"],
                               variable=self.cp_poison_str).pack(fill="x", padx=12)

        ctk.CTkFrame(ctrl, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=8)

        # Progress
        self.cp_progress = ctk.CTkProgressBar(ctrl)
        self.cp_progress.pack(fill="x", padx=12, pady=(0, 2))
        self.cp_progress.set(0)
        self.cp_status = ctk.CTkLabel(ctrl, text="Ready", text_color="gray",
                                       font=ctk.CTkFont(size=11))
        self.cp_status.pack(pady=(0, 4))

        # Action buttons
        ctk.CTkButton(ctrl, text="Apply & Save",
                      fg_color="#c0392b", hover_color="#922b21",
                      font=ctk.CTkFont(weight="bold"),
                      command=self._cp_apply).pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkLabel(ctrl,
                     text="Note: video audio is not preserved.\nRe-add with ffmpeg if needed.",
                     text_color="gray40", font=ctk.CTkFont(size=10),
                     wraplength=240).pack(padx=12)

        # Internal state
        self._cp_source_path: str | None = None
        self._cp_is_video: bool = False

    # ------------------------------------------------------------------
    # Image display helper
    # ------------------------------------------------------------------

    def _show_image(
        self,
        pil_img: Image.Image,
        label: ctk.CTkLabel,
        max_px: int = 400,
    ) -> None:
        display = pil_img.convert("RGB").copy()
        display.thumbnail((max_px, max_px), Image.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=display, dark_image=display, size=display.size
        )
        label.configure(image=ctk_img, text="")
        label._img_ref = ctk_img   # prevent garbage collection

    # ------------------------------------------------------------------
    # Digital tab — actions
    # ------------------------------------------------------------------

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.result_image = None
            self._show_image(self.original_image, self.orig_lbl)
            self.result_lbl.configure(
                image=None,
                text="Protected image will appear here\nafter applying protection",
            )
            if hasattr(self.result_lbl, "_img_ref"):
                self.result_lbl._img_ref = None
            self.d_status.configure(text="Image loaded — click Apply Protection")
            self.d_progress.set(0)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")

    def _on_eps_change(self, value) -> None:
        v = float(value)
        level = "Subtle" if v < 0.035 else "Medium" if v < 0.075 else "Strong"
        self.d_eps_lbl.configure(text=f"{level}  (ε = {v:.3f})")

    def _apply_protection(self) -> None:
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        if self._processing:
            return

        self._processing = True
        epsilon    = self.d_eps.get()
        iterations = QUALITY_ITERS[self.d_quality.get()]
        face_only  = self.d_scope.get() == "Faces Only"

        self.d_progress.set(0)
        self.d_status.configure(text="Loading models…")

        def _run() -> None:
            try:
                engine = self._get_engine()

                def _on_progress(cur: int, tot: int) -> None:
                    def _update() -> None:
                        self.d_progress.set(cur / tot)
                        self.d_status.configure(text=f"Step {cur} / {tot}")
                    self.after(0, _update)

                engine.progress_callback = _on_progress
                self.after(0, lambda: self.d_status.configure(text="Detecting faces…"))

                result = engine.protect_digital(
                    self.original_image,
                    epsilon=epsilon,
                    iterations=iterations,
                    face_only=face_only,
                )
                self.result_image = result

                def _done() -> None:
                    self._show_image(result, self.result_lbl)
                    self.d_status.configure(text="Done!  Protection applied.")
                    self.d_progress.set(1.0)
                    self._processing = False

                self.after(0, _done)

            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.d_status.configure(text="Error — see dialog"))
                self._processing = False

        threading.Thread(target=_run, daemon=True).start()

    def _save_digital(self) -> None:
        if self.result_image is None:
            messagebox.showwarning("Nothing to Save", "Apply protection first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG — lossless, recommended", "*.png"),
                ("JPEG", "*.jpg"),
            ],
        )
        if path:
            kw = {"quality": 95} if path.lower().endswith((".jpg", ".jpeg")) else {}
            self.result_image.save(path, **kw)
            self.d_status.configure(text="Saved!")

    # ------------------------------------------------------------------
    # Pattern tab — actions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Content Protection tab — actions
    # ------------------------------------------------------------------

    def _cp_load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self._cp_source_path = path
            self._cp_is_video = False
            self._show_image(img, self.cp_preview, max_px=480)
            self.cp_file_lbl.configure(text=f"Image: {path.split('/')[-1]}  ({img.size[0]}×{img.size[1]})")
            self.cp_status.configure(text="Image loaded — click Apply & Save")
            self.cp_progress.set(0)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _cp_load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
        )
        if not path:
            return
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {path}")
            return
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Show first frame as preview
        ret, frame = cap.read()
        cap.release()
        if ret:
            pil_frame = Image.fromarray(__import__('cv2').cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._show_image(pil_frame, self.cp_preview, max_px=480)
        self._cp_source_path = path
        self._cp_is_video = True
        self.cp_file_lbl.configure(
            text=f"Video: {path.split('/')[-1]}  {w}×{h} · {total} frames · {fps:.1f} fps"
        )
        self.cp_status.configure(text="Video loaded — click Apply & Save")
        self.cp_progress.set(0)

    def _cp_apply(self) -> None:
        if not self._cp_source_path:
            messagebox.showwarning("No File", "Load an image or video first.")
            return
        if self._processing:
            return

        qr_enabled  = self.cp_qr_enable.get()
        qr_text     = self.cp_qr_text.get().strip() or "© Protected — No AI Training"
        qr_opacity  = self.cp_opacity.get()
        qr_pos      = self.cp_qr_pos.get()
        poison_on   = self.cp_poison_enable.get()
        poison_eps  = {"Subtle": 0.025, "Medium": 0.04, "Strong": 0.065}[self.cp_poison_str.get()]

        if not qr_enabled and not poison_on:
            messagebox.showwarning("Nothing to do", "Enable at least one protection.")
            return

        if self._cp_is_video:
            default_ext = ".mp4"
            ftypes = [("MP4 video", "*.mp4")]
        else:
            default_ext = ".png"
            ftypes = [("PNG", "*.png"), ("JPEG", "*.jpg")]

        out_path = filedialog.asksaveasfilename(defaultextension=default_ext, filetypes=ftypes)
        if not out_path:
            return

        self._processing = True
        self.cp_progress.set(0)
        self.cp_status.configure(text="Loading models…")

        def _run() -> None:
            try:
                engine = self._get_engine()

                if self._cp_is_video:
                    def frame_cb(cur, tot):
                        def _u():
                            self.cp_progress.set(cur / tot)
                            self.cp_status.configure(text=f"Frame {cur} / {tot}")
                        self.after(0, _u)

                    engine.protect_video(
                        self._cp_source_path, out_path,
                        qr_text=qr_text if qr_enabled else None,
                        qr_opacity=qr_opacity,
                        qr_position=qr_pos,
                        add_poison=poison_on,
                        poison_epsilon=poison_eps,
                        poison_iterations=20,
                        frame_callback=frame_cb,
                    )
                else:
                    img = Image.open(self._cp_source_path).convert("RGB")
                    total_steps = (40 if poison_on else 0) + (1 if qr_enabled else 0)
                    done = [0]

                    if poison_on:
                        def prog(cur, tot):
                            def _u():
                                self.cp_progress.set(cur / tot * (40 / total_steps))
                                self.cp_status.configure(text=f"Poisoning… {cur}/{tot}")
                            self.after(0, _u)
                        engine.progress_callback = prog
                        self.after(0, lambda: self.cp_status.configure(text="Applying training poison…"))
                        img = engine.add_training_poison(img, epsilon=poison_eps, iterations=40)

                    if qr_enabled:
                        self.after(0, lambda: self.cp_status.configure(text="Adding QR watermark…"))
                        img = engine.add_qr_watermark(img, qr_text, qr_opacity, qr_pos)

                    kw = {"quality": 95} if out_path.lower().endswith((".jpg", ".jpeg")) else {}
                    img.save(out_path, **kw)

                    # Show result in preview
                    self.after(0, lambda i=img: self._show_image(i, self.cp_preview, max_px=480))

                def _done():
                    self.cp_progress.set(1.0)
                    self.cp_status.configure(text="Saved!")
                    self._processing = False

                self.after(0, _done)

            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.cp_status.configure(text="Error"))
                self._processing = False

        threading.Thread(target=_run, daemon=True).start()

    def _on_shape_change(self, value: str) -> None:
        if value == "Body Patch":
            self.body_hint.pack(padx=16, pady=(4, 0))
        else:
            self.body_hint.pack_forget()

    def _generate_pattern(self) -> None:
        if self._processing:
            return

        self._processing = True
        shape_label = self.p_shape.get()
        shape_key = {"Glasses": "glasses", "Full Patch": "patch", "Body Patch": "body"}[shape_label]
        size      = PATTERN_SIZES[shape_key][self.p_size.get()]
        iters     = QUALITY_ITERS[self.p_quality.get()] * 3   # patterns need more steps

        self.p_progress.set(0)
        self.p_status.configure(text="Loading models…")

        def _run() -> None:
            try:
                engine = self._get_engine()

                def _on_progress(cur: int, tot: int) -> None:
                    def _update() -> None:
                        self.p_progress.set(cur / tot)
                        self.p_status.configure(text=f"Optimising… {cur} / {tot}")
                    self.after(0, _update)

                engine.progress_callback = _on_progress

                if shape_key == "body":
                    result = engine.generate_body_pattern(
                        size=size, iterations=iters
                    )
                else:
                    result = engine.generate_pattern(
                        shape=shape_key, size=size, iterations=iters,
                    )
                self.pattern_image = result

                def _done() -> None:
                    self._show_image(result, self.pattern_lbl, max_px=580)
                    self.p_status.configure(text="Pattern ready — save and print!")
                    self.p_progress.set(1.0)
                    self._processing = False

                self.after(0, _done)

            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.p_status.configure(text="Error — see dialog"))
                self._processing = False

        threading.Thread(target=_run, daemon=True).start()

    def _save_pattern_png(self) -> None:
        if self.pattern_image is None:
            messagebox.showwarning("Nothing to Save", "Generate a pattern first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG with transparency", "*.png")],
        )
        if path:
            self.pattern_image.save(path)
            self.p_status.configure(text="PNG saved!")

    def _save_pattern_pdf(self) -> None:
        if self.pattern_image is None:
            messagebox.showwarning("Nothing to Save", "Generate a pattern first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF — print at 100% scale", "*.pdf")],
        )
        if not path:
            return
        rgb = self.pattern_image.convert("RGB")
        # Body patches print at 30 cm; glasses/other at 15 cm
        cm_wide = BODY_PATCH_PRINT_CM if self.p_shape.get() == "Body Patch" else 15
        target_w = int(cm_wide / 2.54 * 300)
        scale = target_w / rgb.width
        pdf_img = rgb.resize((target_w, int(rgb.height * scale)), Image.LANCZOS)
        pdf_img.save(path, "PDF", resolution=300)
        self.p_status.configure(text=f"PDF saved — print at 100% scale ({cm_wide} cm wide)!")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceShieldApp()
    app.mainloop()
