# Local Dataset Root Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local project data reads work after the dataset root was moved to `E:\CodeWorking\datasets\AnJuXiaoBaoKWS`.

**Architecture:** Replace only local Windows dataset-root references in project configuration and utilities, and localize server-generated dataset manifests that must open WAV files on this PC. Keep server shell scripts and model artifacts unchanged unless they refer to the local Windows root.

**Tech Stack:** PowerShell, Python/JSONL manifests, YAML configuration, WeKWS dataset lists.

---

### Task 1: Replace local project data-root references

**Files:**
- Modify: `configs/data/*.yaml`
- Modify: `scripts/*.cmd`
- Modify: `src/anju_kws/**/*.py`
- Modify: `data/prepared_pretrain_posbalanced_mid_20260509/**/*.{scp,jsonl}`

- [x] Scan for the old root `E:\CodeWorking\Dataset\AnJuXiaoBaoKWS`.
- [x] Replace it with `E:\CodeWorking\datasets\AnJuXiaoBaoKWS`, preserving slash style already present in each file.
- [x] Re-scan the modified project scopes and confirm there are no remaining old-root references.

### Task 2: Localize downloaded server dataset manifests

**Files:**
- Modify: `E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\**\*.jsonl`
- Modify: `E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\**\*.scp`

- [x] Identify files whose audio paths begin with `/home/clm/datasets/AnJuXiaoBaoKWS`.
- [x] Replace the server dataset root with `E:/CodeWorking/datasets/AnJuXiaoBaoKWS`.
- [x] Verify sampled `wav` or `path` values point to existing local WAV files.

### Task 3: Verify active data inputs

**Files:**
- Verify: `E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\anju_xiaobao_cosyvoice3_cleaned_farfield_20260521\manifest.jsonl`
- Verify: `E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\prepared_wekws_farfield_main_20260521\train\data.list`
- Verify: `E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\prepared_wekws_farfield_mined_20260521\train\data.list`

- [x] Parse representative JSONL records and test that their input audio paths exist.
- [x] Compile modified Python utilities with `python -m py_compile`.
- [x] Report any historical configuration that still targets a dataset no longer retained locally.

### Implementation Note

`prepared_wekws_farfield_mined_20260521` also contained 1600 repeated training rows derived from 16 hard-negative windows whose original paths pointed into a server run directory. The local data repository retained the three source continuous recordings, so the 16 five-second windows were rebuilt under `mined_hard_negative_wav/` and the mined train lists were redirected there. This keeps the local prepared dataset self-contained.
