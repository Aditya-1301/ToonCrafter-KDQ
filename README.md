# ToonCrafter-KDQ: LoRA Adaptation, Knowledge Distillation & PTQ for Practical 2D Animation Interpolation

**A research fork of [ToonCrafter](https://github.com/Doubiiu/ToonCrafter)** focused on **parameter-efficient domain adaptation (LoRA)**, **model compression via Knowledge Distillation (KD)**, and **Post-Training Quantization (PTQ)**—with consistent **evaluation** and **report generation**.

* Upstream paper & project page: ToonCrafter (SIGGRAPH Asia 2024) ([GitHub](https://github.com/Doubiiu/ToonCrafter), [arXiv](https://arxiv.org/abs/2405.17933))
* Model card/checkpoint info: Hugging Face (Doubiiu/ToonCrafter) ([Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt))
* FP16 Quantised card/checkpoint info: Hugging Face (Aditya-1301/ToonCrafter_FP16_Quantized) ([Hugging Face](https://huggingface.co/Aditya-1301/ToonCrafter_FP16_Quantized/blob/main/quantized_model_fp16.ckpt))
* Dataset used here: **ATD-12K** from AnimeInterp (CVPR’21) ([Kaggle](https://www.kaggle.com/datasets/marafey/atd-12-dataset), [GitHub](https://github.com/lisiyao21/AnimeInterp), [arXiv](https://arxiv.org/abs/2104.02495))

---

## ✨ What’s new in this fork?

1. **LoRA fine-tuning** on ATD-12K

   * Adapters **only in spatial cross-attention (attn2)**: `{to_q, to_k, to_v, to_out}`.
   * Rank=16, α=16, dropout=0.1.
   * **Freeze map:** base UNet, VAE (enc/dec), text encoder **not in optimizer** (effectively frozen).
   * Effective trainables: **≈3.385M / 1.442B (≈0.235%)**.
   * Inference policy: **LoRA scale `s = 0.15`** (stable sweet-spot on ATD-12K).

2. **Knowledge Distillation (KD)** to a smaller student

   * Two regimes:

     * **Baseline teacher (no LoRA):** image-space supervision **(L1 + LPIPS)**.
     * **LoRA teacher:** image + **latent SmoothL1** + **noise MSE** at a fixed diffusion step $t$.
   * Teacher frozen; **student VAE frozen & copied**; optimizer covers **the rest of the student**.

3. **Post-Training Quantization (PTQ)** (FP16/BF16)

   * **UNet-only** casting to FP16/BF16 (VAE/text kept FP32).
   * In our runs: **≈27% lower peak VRAM** and **≈2.7× faster/frame** vs FP32 baseline.

4. **End-to-end evaluation + reporting**

   * Scripted **PSNR / SSIM / LPIPS** for baseline, LoRA, and quantized models.
   * One command to aggregate CSVs into paper-ready tables.

---

## 📦 Repository additions (this fork)

**Training / Compression**

* `lora_script_3-2.py` — LoRA fine-tuning on ATD-12K
* `distill_student_kd_combined.py` — KD training loop (teacher → student)
* `ptq.py` — FP32 → FP16/BF16 conversion (UNet-only) & save

**Evaluation**

* `metric_evaluation/evaluate_baseline_metrics.py` — baseline FP32
* `metric_evaluation/baseline_evaluation_metrics_lora_new.py` — LoRA/teacher
* `metric_evaluation/evaluate_quantized.py` — quantized models
* `evaluate_lora_final.py` — convenience evaluator for LoRA + scale

**Utilities**

* `custom_utils/datasets.py` — ATD-12K dataset loader
* `custom_utils/debugging_utils.py` — NaN/shape guards, dtype checks
* `check_latent_stats.py` — latent variance/std checks across denoising
* `generate_final_report_data.py` — merges result CSVs → tables/Markdown

---

## 🧠 Method overview

### LoRA (domain adaptation)

**Objective (weighted sum):**

```text
L_LoRA = λ_ε * || ε̂ − ε ||_1  +  λ_pixel * LPIPS(x̂, x)
```

* First term is **L1 on predicted vs. true diffusion noise** at the sampled timestep $t$ (**ε-space**, not latent L1).
* LPIPS computed on decoded RGBs (VGG backbone).
* Trainables: LoRA adapters in **attn2** only.
* Implementation note: if CLIP vision dim ≠ image-proj in, a tiny **Linear CLIP adapter** is created and included in the optimizer; otherwise only LoRA params get gradients.
* Recommended inference: **`--lora_scale 0.15`**.

### Knowledge Distillation (compression)

We use **two KD regimes**:

**(A) Baseline teacher (no LoRA)** — teacher output via normal sampling (e.g., 20 DDIM steps).

```text
L_KD(base) = λ1 * || x̂ − x_T ||_1  +  λ_lpips * LPIPS(x̂, x_T)
```

**(B) LoRA teacher** — add single-timestep supervision in latent + noise spaces (same $t$ for teacher/student):

```text
L_KD = λ1 * || x̂ − x_T ||_1
     + λ_lpips * LPIPS(x̂, x_T)
     + λ_z * SmoothL1(ẑ, z_T)
     + λ_ε * || ε̂ − ε_T ||_2^2
```

* Weights used: `λ1=1.0, λ_lpips=0.5, λ_z=0.5, λ_ε=0.25`.
* Teacher **fully frozen**; **student VAE frozen & copied**; optimizer covers the **rest of student** (UNet + conditioning/text stack).
* For LoRA teachers, set **`--lora_scale 0.15`** for stability.

### Post-Training Quantization (FP16/BF16)

* **UNet-only** casting via `.half()` or BF16; VAE/text left FP32 for stability & simplicity.
* In our tests on ATD-12K: **≈27% lower peak VRAM** and **≈2.7×** speedup/frame vs FP32 baseline, with the expected quality drop.

---

## 🛠️ Setup

```bash
# Python as in upstream (3.8 range); create env and install
conda create -n tooncrafter python=3.8.5
conda activate tooncrafter
pip install -r requirements.txt
```

**Checkpoints layout**

```
checkpoints/tooncrafter_512_interp_v1/model.ckpt
```

---

## 📚 Dataset (ATD-12K)

Structure expected (triplets I0, I1(gt), I2); see `custom_utils/datasets.py`:

```
atd12k_dataset/
├─ train/<clip_id>/frame1.png
│               /frame2.png   # GT middle
│               /frame3.png
├─ val/...
└─ test/...
```

---

## ▶️ Inference (baseline ToonCrafter)

```bash
python scripts/inference.py \
  --config    configs/inference_512_v1.0.yaml \
  --ckpt_path checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --start_img path/to/frame1.png \
  --end_img   path/to/frame3.png \
  --outdir    outputs/baseline
```

---

## 🏋️ LoRA fine-tuning

**Train**

```bash
python lora_script_3-2.py \
  --dataset_path /path/to/atd12k_dataset \
  --ckpt_path    checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --config       configs/inference_512_v1.0.yaml \
  --output_dir   outputs_lora/full_run \
  --epochs 10 --bs 4 --seed 42
# LoRA defaults: rank=16, alpha=16, dropout=0.1
```

**Evaluate LoRA (and control scale)**

```bash
python evaluate_lora_final.py \
  --dataset_path /path/to/atd12k_dataset \
  --config       configs/inference_512_v1.0.yaml \
  --ckpt_path    checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --lora_dir     outputs_lora/full_run/lora_best \
  --lora_scale   0.15 \
  --outdir       outputs_lora/eval_s015
```

> Tip: Avoid `lora_scale=1.0` (variance collapse & grey/noisy frames).

---

## 🧪 Knowledge Distillation

**Baseline-teacher KD (image-space only)**

```bash
python -u distill_student_kd_combined.py \
  --dataset_path     /path/to/atd12k_dataset \
  --teacher_config   configs/inference_512_v1.0.yaml \
  --teacher_ckpt     checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --student_config   configs/student_inference_512_v2.0.yaml \
  --output_dir       outputs_kd/run_base_teacher \
  --epochs 4 --bs 4 --seed 123
```

**LoRA-teacher KD (image + latent + noise)**

```bash
python -u distill_student_kd_combined.py \
  --dataset_path      /path/to/atd12k_dataset \
  --teacher_config    configs/inference_512_v1.0.yaml \
  --teacher_ckpt      checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --teacher_lora_dir  outputs_lora/full_run/lora_best \
  --lora_scale        0.15 \
  --student_config    configs/student_inference_512_v2.0.yaml \
  --output_dir        outputs_kd/run_lora_teacher \
  --epochs 4 --bs 4 --seed 123
```

> In our experiments, the student learned image-space metrics but collapsed at inference when distilled from a LoRA teacher; try a clean baseline teacher first.

---

## 🔻 Post-Training Quantization (PTQ)

**Convert FP32 → FP16 or BF16 (UNet-only)**

```bash
python ptq.py \
  --src_ckpt checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --dst_ckpt final_results/quantized/quantized_model_fp16.ckpt \
  --dtype    fp16    # or: bf16
```

**Evaluate quantized model**

```bash
python metric_evaluation/evaluate_quantized.py \
  --dataset_path /path/to/atd12k_dataset \
  --config       configs/inference_512_v1.0.yaml \
  --ckpt_path    final_results/quantized/quantized_model_fp16.ckpt \
  --dtype        fp16 \
  --out_csv      results_fp16.csv
```

> Observed on our setup: **≈27% lower peak VRAM** and **≈2.7×** faster per-frame vs FP32 baseline.

---

## 📏 Metric evaluation (PSNR / SSIM / LPIPS)

**Baseline (FP32)**

```bash
python metric_evaluation/evaluate_baseline_metrics.py \
  --dataset_path /path/to/atd12k_dataset \
  --config       configs/inference_512_v1.0.yaml \
  --ckpt_path    checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --out_csv      results_baseline.csv
```

**LoRA**

```bash
python metric_evaluation/baseline_evaluation_metrics_lora_new.py \
  --dataset_path /path/to/atd12k_dataset \
  --config       configs/inference_512_v1.0.yaml \
  --ckpt_path    checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --lora_dir     outputs_lora/full_run/lora_best \
  --lora_scale   0.15 \
  --out_csv      results_lora_s015.csv
```

**PTQ (FP16/BF16)**

```bash
python metric_evaluation/evaluate_quantized.py \
  --dataset_path /path/to/atd12k_dataset \
  --config       configs/inference_512_v1.0.yaml \
  --ckpt_path    final_results/quantized/quantized_model_fp16.ckpt \
  --dtype        fp16 \
  --out_csv      results_fp16.csv
```

---

## 📊 Results summary (ATD-12K, 2k test triplets)

| Model                 | PSNR ↑ | SSIM ↑ | LPIPS (VGG) ↓ |
| --------------------- | :----: | :----: | :-----------: |
| **Baseline (FP32)**   |  24.07 | 0.8632 |     0.1314    |
| **LoRA (scale 0.15)** |  23.24 | 0.8086 |     0.2269    |
| **PTQ (FP16)**        |  20.42 | 0.6783 |     0.3203    |

**Efficiency (illustrative, same setup):**

| Model                | Ckpt Size | Peak VRAM | ms / frame |
| -------------------- | --------- | --------- | ---------- |
| Baseline (FP32)      | 9.78 GB   | 19.13 GB  | 1971.8     |
| LoRA\@0.15 (+base)   | +25 MB    | 19.16 GB  | 2112.1     |
| Student (KD, failed) | 5.31 GB   | 14.65 GB  | 730.1      |
| PTQ (FP16)           | 7.10 GB   | 13.95 GB  | **726.8**  |

> Numbers are from our thesis runs on the same hardware; your mileage may vary.

---

## 🧪 Debugging & stability tools

* `check_latent_stats.py` — tracks latent std/variance to catch **variance collapse**.
* `custom_utils/debugging_utils.py` — NaN/dtype/shape guards and logging hooks.

---

## 📄 Reproducing paper tables

```bash
python generate_final_report_data.py \
  --inputs results_baseline.csv results_lora_s015.csv results_fp16.csv \
  --out    tables/final_results.md
```

---

## ⚠️ Known limitations / guidance

* **LoRA scaling matters.** `lora_scale=1.0` often collapses; **0.1–0.35** is a safer grid, **0.15** best in our tests.
* **KD inherits instability.** If a teacher is even slightly unstable, a student can amplify it; start with a **clean baseline teacher**.
* **PTQ is the pragmatic baseline.** Expect a quality drop; consider mixed-precision/QAT for further trade-off.

---

## 📜 Citation

Please cite the original ToonCrafter and (if relevant) this thesis/report.

```bibtex
@article{xing2024tooncrafter,
  title={ToonCrafter: Generative Cartoon Interpolation},
  author={Xing, Jinbo and Liu, Hanyuan and Xia, Menghan and Zhang, Yong and Wang, Xintao and Shan, Ying and Wong, Tien-Tsin},
  journal={arXiv:2405.17933},
  year={2024}
}
```

---

## 🙏 Acknowledgements

This fork builds directly on **ToonCrafter**. See upstream for credits and license.

---

### Appendix: exact LoRA/KD settings used in our runs (for reviewers)

* **LoRA adapters:** attn2 `{to_q, to_k, to_v, to_out.0}`, rank=16, α=16, dropout=0.1.
* **Trainable parameters:** ≈3.385M / 1.442B (≈0.235%).
* **LoRA inference scale:** **0.15**.
* **KD loss weights:** `λ1=1.0, λ_lpips=0.5, λ_z=0.5, λ_ε=0.25`.
* **Validation metrics:** PSNR↑, SSIM↑, LPIPS(VGG)↓ on ATD-12K test (2k triplets).

---

### License

This fork follows the upstream ToonCrafter license (see `LICENSE` here and upstream).

### Changelog (fork)

* **2025-08-21**: README updated to match final LoRA/KD/PTQ design (ε-L1 + LPIPS for LoRA, two-regime KD, UNet-only PTQ; speed/VRAM numbers aligned).
