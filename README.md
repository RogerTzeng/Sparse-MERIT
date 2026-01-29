# Sparse MERIT: Joint Learning using Mixture-of-Expert-Based Representation for Speech Enhancement and Robust Emotion Recognition

Official repository for the TASLP submission:

**“Joint Learning using Mixture-of-Expert-Based Representation for Speech Enhancement and Robust Emotion Recognition”**  
(aka **Sparse MERIT**)

> ⚠️ **Status:** Initial repository setup. Code and pretrained checkpoints will be released/organized progressively.  
> This README provides the intended structure, usage plan, and reproduction notes.

---

## Overview

This work studies **joint learning** of:
- **Speech Enhancement (SE)**
- **Speech Emotion Recognition (SER)**

We propose **Sparse MERIT**, a Mixture-of-Experts (MoE) representation framework that performs **frame-wise sparse routing** to encourage task-adaptive specialization while reducing interference between SE and SER objectives.

Key ideas:
- A shared SSL backbone provides frame-level representations.
- Task-specific routing selects experts (Top-1 in Sparse MERIT) to produce task-adaptive features.
- Joint training optimizes SE and SER objectives while mitigating negative transfer.

---

## What’s in this repo (planned)

