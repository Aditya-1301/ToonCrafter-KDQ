# ToonCrafter-KDQ: LoRA Adaptation, Knowledge Distillation & PTQ for Practical 2D Animation Interpolation

**A research fork of [ToonCrafter](https://github.com/Doubiiu/ToonCrafter)** focused on **parameter-efficient domain adaptation (LoRA)**, **model compression via Knowledge Distillation (KD)**, and **Post-Training Quantization (PTQ)**—with consistent **metric evaluation** and **report generation** tools.

* Upstream paper & project page: ToonCrafter (SIGGRAPH Asia 2024) ([GitHub](https://github.com/Doubiiu/ToonCrafter), [arXiv](https://arxiv.org/abs/2405.17933))
* Model card / checkpoint info: Hugging Face (ToonCrafter) ([Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt))
* Dataset used here: **ATD-12K** from AnimeInterp (CVPR’21) ([Kaggle](https://www.kaggle.com/datasets/marafey/atd-12-dataset), [GitHub](https://github.com/lisiyao21/AnimeInterp), [arXiv](https://arxiv.org/abs/2104.02495))

---

## ✨ What’s new in this fork?

This repository adds a **practical pipeline** on top of the original ToonCrafter:

1. **LoRA fine-tuning** of ToonCrafter on ATD-12K

   * Rank=16, α=16, dropout=0.1; adapters injected only in **spatial cross-attention (attn2)** on projections **to\_q / to\_k / to\_v / to\_out**.
   * **Freeze map:** VAE (enc/dec), text encoder, **all temporal blocks** of the 3D-UNet are **frozen**.
   * Effective trainables: **≈3.385M / 1.442B (0.235%)**.
   * Inference stability via **LoRA scale s=0.15**.&#x20;

2. **Knowledge Distillation (KD) pipeline** from the LoRA-adapted “teacher” to a smaller “student” UNet

   * Multi-component loss supervising **RGB, latent, and noise** predictions (details below).
   * Findings: Instability can transfer from teacher to student; the tuned student in this study collapsed at inference despite promising early metrics (documented openly below).&#x20;

3. **Post-Training Quantization (PTQ)** to FP16 (zero-retraining) with evaluation scripts

   * Delivers **\~27% VRAM reduction** and **≈3× faster/frame** vs. FP32 baseline in our tests on ATD-12K, with acceptable quality trade-offs.&#x20;

4. **End-to-end evaluation & reporting**

   * Scripted **PSNR / SSIM / LPIPS** evaluation for **baseline, LoRA, and quantized** models and a single command to **generate final tables** for papers/theses.&#x20;

---

## 📦 Repository additions (your contributions)

* **Training / Compression**

  * `lora_script_3-2.py` — LoRA fine-tuning on ATD-12K
  * `distill_student_kd_combined.py` — KD training loop (teacher→student)
  * `ptq.py` — FP32→FP16/BF16 conversion & serialization

* **Evaluation**

  * `metric_evaluation/baseline_evaluation_metrics_lora_new.py` — eval for LoRA/teacher
  * `metric_evaluation/evaluate_baseline_metrics.py` — eval for baseline FP32
  * `metric_evaluation/evaluate_quantized.py` — eval for quantized models
  * `evaluate_lora_final.py` — convenience evaluator for LoRA + scale

* **Utilities**

  * `custom_utils/datasets.py` — ATD-12K loader utilities
  * `custom_utils/debugging_utils.py` — stability/NaN guards; latent stats hooks
  * `check_latent_stats.py` — variance checks across denoising steps
  * `generate_final_report_data.py` — aggregate metrics → CSV/tables for papers

These scripts and their design choices (losses, freeze map, scaling, splits, logging) are documented in the accompanying thesis; this README condenses the operational highlights.&#x20;

---

## 🧠 Method overview

### LoRA (domain adaptation)

**Loss (weighted sum):**

```
L_LoRA = λ_latent * || z_pred - z_gt ||_1 + λ_pixel * LPIPS( x_hat, x )
```

* `z_pred, z_gt`: VAE-latent predictions/targets; `x_hat, x`: decoded RGBs.
* LPIPS in \[0,1]; latent/LPIPS terms normalized appropriately.
* **Trainable params:** LoRA adapters only (attn2: to\_q, to\_k, to\_v, to\_out).
* **Recommended inference:** `lora_scale = 0.15` (stability sweet-spot).&#x20;

### Knowledge Distillation (compression)

**Student loss (weighted sum):**

```
L_KD = λ1 * || x_hat - x_T ||_1
     + λ_lpips * LPIPS( x_hat, x_T )
     + λ_z * SmoothL1( z_hat, z_T )
     + λ_ε * || ε_hat - ε_T ||_2^2
```

* Teacher outputs at matching timestep `t`; VAE frozen/copy; student UNet trainable.
* Weights used in experiments: `λ1=1.0, λ_lpips=0.5, λ_z=0.5, λ_ε=0.25`.
* **Caveat:** A teacher with inference-time instability can transfer failure modes to the student (observed collapse even when early metrics improved).&#x20;

### Post-Training Quantization (FP16)

* Simple `.half()` conversion with safety casts for sensitive modules (e.g., time embeddings), then evaluated with specialized loaders.
* **Why here?** Diffusion models tolerate mixed/half precision reasonably well with modest quality drop, offering a low-effort deployment path.&#x20;

---

## 🛠️ Setup

> The base environment follows upstream ToonCrafter; install their deps first.

```bash
# Python 3.8 recommended by upstream; create env and install
conda create -n tooncrafter python=3.8.5
conda activate tooncrafter
pip install -r requirements.txt
```

**Checkpoints**
Place the upstream checkpoint at:

```
checkpoints/tooncrafter_512_interp_v1/model.ckpt
```

---

## 📚 Dataset (ATD-12K)

Download ATD-12K from AnimeInterp and arrange as triplets (I0, I1(gt), I2). Typical structure: ([GitHub][5], [ar5iv][6])

```
atd12k_dataset/
├── train/
│   ├── <clip_id>/frame1.png
│   ├── <clip_id>/frame2.png  # GT middle
│   └── <clip_id>/frame3.png
├── val/...
└── test/...
```

> Exact expectations (filenames, splits) are implemented in `custom_utils/datasets.py`. Adjust paths with `--dataset_path`.&#x20;

---

## ▶️ Inference (baseline ToonCrafter)

```bash
# Two-keyframe interpolation using upstream configs
python scripts/inference.py \
  --config configs/inference_512_v1.0.yaml \
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
# LoRA default: rank=16, alpha=16, dropout=0.1 (as set in this script)
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

> **Tip:** Avoid `lora_scale=1.0`—we observed latent variance collapse and grey/noisy frames at full strength; `0.15` was consistently stable.&#x20;

---

## 🧪 Knowledge Distillation

**Train student from LoRA teacher**

```bash
python -u distill_student_kd_combined.py \
  --dataset_path      /path/to/atd12k_dataset \
  --teacher_config    configs/inference_512_v1.0.yaml \
  --teacher_ckpt      checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --teacher_lora_dir  outputs_lora/full_run/lora_best \
  --student_config    configs/student_inference_512_v2.0.yaml \
  --output_dir        outputs_kd/run01 \
  --epochs 4 --bs 4 --seed 123
```

> **Known issue (documented):** Despite promising early metrics, our distilled student **collapsed at inference** (temporal instability/variance collapse). If reproducing, prefer a **clean baseline teacher** (no LoRA) for first KD experiments.&#x20;

---

## 🔻 Post-Training Quantization (PTQ)

**Convert FP32 → FP16 or BF16**

```bash
# Example: create FP16 checkpoint
python ptq.py \
  --src_ckpt  checkpoints/tooncrafter_512_interp_v1/model.ckpt \
  --dst_ckpt  final_results/quantized/quantized_model_fp16.ckpt \
  --dtype     fp16
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

> We observed **\~27% lower peak VRAM** and **≈3× speed-up** per frame versus FP32 baseline on ATD-12K, with expected quality drop.&#x20;

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

> Numbers taken from the accompanying thesis runs on our hardware; exact values may vary by GPU / I/O.&#x20;

---

## 🧪 Debugging & stability tools

* `check_latent_stats.py` — track latent std across denoising; detects **variance collapse** (a strong indicator of LoRA scale too high or unstable teachers).
* `custom_utils/debugging_utils.py` — hooks to clamp/guard, detect NaNs, and log shapes/dtypes at critical points.&#x20;

---

## 📄 Reproducing paper tables

```bash
python generate_final_report_data.py \
  --inputs results_baseline.csv results_lora_s015.csv results_fp16.csv \
  --out    tables/final_results.md
```

> Produces clean, copy-pasteable markdown/CSV for reports and slides.&#x20;

---

## ⚠️ Known limitations / guidance

* **LoRA scaling matters:** `lora_scale=1.0` produced grey/noisy collapse; keep **0.15** for stable outputs (or grid-search 0.1–0.35).&#x20;
* **KD inherits instability:** If your teacher has subtle artefacts or variance issues, the student is likely to **amplify** them at inference. Try **distilling from a clean baseline** before distilling LoRA.&#x20;
* **PTQ is the pragmatic baseline:** Expect quality drop, but it’s often the best “one afternoon” path to get ToonCrafter onto smaller GPUs. Consider **QAT / mixed-precision** for future work.&#x20;

---

## 🔗 References

* ToonCrafter (paper, code). ([arXiv](https://arxiv.org/abs/2405.17933), [GitHub](https://github.com/Doubiiu/ToonCrafter))
* AnimeInterp + ATD-12K dataset. ([Kaggle](https://www.kaggle.com/datasets/marafey/atd-12-dataset), [GitHub](https://github.com/lisiyao21/AnimeInterp), [arXiv](https://arxiv.org/abs/2104.02495))
* ToonCrafter HF model card. ([Hugging Face](https://huggingface.co/Doubiiu/ToonCrafter/blob/main/model.ckpt))

---

## 📜 Citation

If this fork or its analysis was useful, please cite the original ToonCrafter and the thesis/report behind this repository.

<details>
<summary>BibTeX (ToonCrafter)</summary>

```bibtex
@article{xing2024tooncrafter,
  title={ToonCrafter: Generative Cartoon Interpolation},
  author={Xing, Jinbo and Liu, Hanyuan and Xia, Menghan and Zhang, Yong and Wang, Xintao and Shan, Ying and Wong, Tien-Tsin},
  journal={arXiv:2405.17933},
  year={2024}
}
```

</details>

---

## 🙏 Acknowledgements

This work builds directly on **ToonCrafter**; please see their repo for credits and license. ([GitHub](https://github.com/Doubiiu/ToonCrafter))

---

### Appendix: exact LoRA/KD settings used in our runs (for reviewers)

* **LoRA adapters:** attn2 `{to_q, to_k, to_v, to_out}`, rank=16, α=16, dropout=0.1; **VAE/text/temporal blocks frozen**.
* **Trainable parameters:** ≈3.385M / 1.442B = **0.235%**.
* **LoRA inference scale:** **0.15**.
* **KD loss weights:** `λ1=1.0, λ_lpips=0.5, λ_z=0.5, λ_ε=0.25`.
* **Metrics:** PSNR↑, SSIM↑, LPIPS(VGG)↓ on ATD-12K test 2k.&#x20;

---

**Note on upstream context & dataset**
ToonCrafter is a generative, dual-reference 3D U-Net latent diffusion model that synthesizes interpolations from two keyframes (and optional text/sketch conditioning). It was trained for 16-frame 512×320 outputs; see upstream paper/card for details. Our work adapts it to ATD-12K triplets and studies stability and compression on that dataset. ([arXiv][3], [Hugging Face][4])

---

### License

This fork follows the licensing terms of the upstream ToonCrafter repository; see `LICENSE` in this repo and the upstream repo for details. ([GitHub](https://github.com/Doubiiu/ToonCrafter))

---

### Changelog (fork)

* **2025-08-21**: Public README overhaul; clarified LoRA/KD/PTQ usage and added evaluation & reporting commands (this doc).
* Earlier: Added LoRA training script, KD training script, PTQ conversion & evaluation utilities, dataset/debugging helpers.&#x20;