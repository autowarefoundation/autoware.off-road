# End-to-End Off-Road Autonomy

This project focuses on building an end-to-end perception pipeline for off-road autonomous vehicles. Our primary goal is to enable **robust and scalable vision models** for challenging, unstructured environments using **self-supervised learning** and **efficient transformer architectures** like SegFormer.

---

## 🚜 Project Overview

Off-road autonomy poses unique challenges in perception due to the lack of structure, sparse datasets, and dynamic environments. Our approach focuses on **object segmentation** as the foundation for reliable perception and planning.

We hypothesize that **attention-based and self-supervised learning** methods can unlock strong performance even with **limited annotated data**, by leveraging inductive biases from transformers and multi-task learning (e.g., depth, segmentation, occupancy flow).

---

## 📦 Tech Stack

| Component        | Role                                                                                      |
|------------------|-------------------------------------------------------------------------------------------|
| **PyTorch + PyTorch Lightning** | Simplified model development, training loop abstraction, and native CUDA integration. |
| **Ray Train & Tune**             | Scalable, distributed training + hyperparameter tuning across GPUs/nodes. Easy to monitor cluster usage. |
| **MLflow**                      | Centralized tracking of experiments, metrics, models, and reproducibility. |
| **Hydra**                        | Configuration management system for flexible experiment workflows (YAML, CLI, Env). |

---

## 🔍 Why SegFormer?

SegFormer is chosen as our baseline architecture because:

- Combines **transformer encoders (MiT)** with **MLP decoders**, making it lightweight yet powerful.
- Captures **multi-scale spatial context**, which ViTs struggle with.
- No fixed positional encodings → improves generalization across unseen terrain.
- Performs well on small or synthetic datasets, reducing overfitting risk.

> Ideal for **self-supervised learning setups** in segmentation, depth estimation, and occupancy prediction.

---

## 🦢 Why Goose Dataset?

- **15,000+ high-resolution off-road images** with pixel-wise labels.
- Contains unstructured elements like **rocks, trees, trails, bushes**, unlike urban datasets (Cityscapes, nuScenes).
- Designed for **outdoor and rough terrain segmentation** tasks.
- Enables training models that generalize to **non-urban, forested, or rugged environments**.

---

## ✅ Current Status

- ✅ Tech stack setup complete (Lightning, Hydra, Ray, MLflow)
- ✅ SegFormer integration and training setup initialized
- 🐞 Currently debugging training pipeline issues
- 📊 Next Steps: Benchmark SegFormer on Goose Dataset vs private working group models

---

## 🧪 Planned Experiments

- Self-supervised pretraining + finetuning
- Multitask learning: segmentation + depth + occupancy flow
- Evaluation on:
  - Standard segmentation metrics (IoU, pixel accuracy)
  - Real-world deployment robustness
  - Reliability across lighting, season, and terrain diversity


---

## 🤝 Contributing

This is a work in progress. If you're interested in collaborating on off-road autonomy, feel free to reach out or open an issue.

