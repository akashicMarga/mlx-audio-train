"""GRPO post-training (Pipeline 3) for Qwen3-TTS.

See docs/grpo_post_training.md. Modules:
  rollout.py   — on-policy cb0 sampler + shared teacher-forced forward + decode
  rewards.py   — programmatic rewards (CER, length guard) → group advantages   [todo]
  ../losses/grpo_loss.py — advantage-weighted PG + k3 KL + optional SFT mixin   [todo]
"""
