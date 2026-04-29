# DOCUMENT.md

<!--
This file lives in the root of every forked repo.
Fill it in as you go. Do not reconstruct it after the fact.
Keep entries factual and brief. The audience is a future person
reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

<!--
Copy this information from the upstream repo's README and paper.
"Paired or unpaired" refers to whether the model assumes paired training data.
If the model is domain-specific to virtual staining, note the exact staining task (e.g. H&E to HER2 IHC).
-->

- **Model name:** CycleGAN and pix2pix in PyTorch
- **Upstream repo URL:** https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- **Fork URL:** https://github.com/InViLabVirtualStainingBenchmark/pytorch-CycleGAN-and-pix2pix
- **Upstream last commit date:** Aug 6, 2025
- **Paper / citation:** 
    - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)
    - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004)
- **Paired or unpaired assumption:** Both
- **Intended staining task (if domain-specific):**

---

## Environment Claimed by Authors

<!--
Record exactly what the authors say in their README or requirements file.
Do not adjust or interpret -- copy their stated versions.
"Requirements file present" should note the filename if it exists.
If no version is specified for Python or PyTorch, write "not specified".
-->

- **Python version:** 3.11
- **PyTorch version:** 2.4.0
- **CUDA version:** 13.1
- **Installation method:** conda
- **Requirements file present:** environment.yml
- **Pretrained weights available:** yes 
- **Pretrained weights notes:** Downloaded via shell scripts from authors' server. Low rot risk.
<!-- Where are they hosted? Are they behind a login? Is the link likely to rot (GDrive, Dropbox, personal server)? -->

---

## Environment Actually Used

<!--
Record the environment you actually created and tested in.
If you deviated from what the authors specified, briefly note why (e.g. "authors' version not compatible with CUDA 12.1").
Conda env name should follow the convention: the model's short name.
-->

- **Python version:** 3.11
- **PyTorch version:** 2.4.0
- **CUDA version:** 12.1
- **Conda environment name:** pytorch-img2img
- **Date tested:** 23-04-2026
- **Hardware:** RTX 4090, WSL2 on Windows 11

---

## Installation

<!--
Follow the authors' README exactly before making any changes.
Record the commands you ran in order.
If an error occurred, paste the key line of the error (not the full traceback) and then record the fix.
If installation succeeded without issues, write "No issues."
-->

### Commands Run

```bash
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix
```

### Issues and Fixes

<!--
Format: problem encountered -> fix applied.
If no issues, write "None."
-->

| Issue | Fix Applied |
| --- | --- |
|  |  |

### GPU Confirmation

<!--
Paste the output of the check below so there is proof the GPU was visible.
Command: python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
-->

```
True NVIDIA GeForce RTX 4090
```

---

## Dataset Preparation

<!--
Record how the dataset was prepared for this specific model.
"Format expected" means what folder layout or file structure the model's data loader assumes
(e.g. side-by-side paired images, separate A/B folders, CSV manifest, etc.).
"Conversion applied" means any script or command you ran to reformat the standard BCI/MIST-HER2
download into the format this model needs.
If no conversion was needed, write "None -- dataset used as downloaded."
-->

- **Dataset used:** BCI, MIST-HER2
- **Format expected by model:** Separate A/B folders (trainA, trainB, testA, testB)
- **Conversion applied:** reused BCI-cut-smoke dataset, format is identical
    
    ```bash
    # paste conversion command(s) here if any
    ```
    
- **Final folder layout used:**
    
    ```
    datasets/BCI-cut-smoke/
        trainA/   <- H&E (200 train, 20 test)
        trainB/   <- IHC (200 train, 20 test)
    datasets/MIST-HER2-cut-smoke/
        trainA/   <- H&E (200 train, 20 test)
        trainB/   <- IHC (200 train, 20 test)
    ```
    
- **Number of images used for smoke test (train / test):**

---

## Pretrained Weights

<!--
Only fill this section if pretrained weights exist.
Record the exact download source. Flag any link that is not on a stable host
(Zenodo and HuggingFace are stable; Google Drive, Dropbox, and personal servers are at risk).
Record where you placed the weights relative to the repo root.
-->

- **Download source URL:**
- **Host stability:** stable (Zenodo / HuggingFace) / at-risk (GDrive / Dropbox / personal server) / N/A
- **Weights placed at (relative path):**
- **Size on disk:**

---

## Inference Smoke Test

<!--
Run inference before training if pretrained weights are available -- it is faster
and confirms the code path works independently of the training loop.
Use 10-20 images from the BCI or MIST-HER2 test split.
"Visual check" is a qualitative sanity check only -- not a metric.
Valid outcomes: "images look like expected domain", "blank/grey output", "wrong resolution", "file not written".
-->

- **Script / command run:**
    
    ```bash
    # paste exact command here
    ```
    
- **Output folder:**
- **Number of output images produced:**
- **Output image dimensions:**
- **Visual check result:**
- **Time to run (approx):**
- **Errors or warnings during inference:**
<!-- "None" if clean. Otherwise paste the key error line. -->

---

## Training Smoke Test

<!--
Run training for 5 epochs minimum. The goal is a clean exit, not a useful model.
Use the smallest viable batch size and the model's default resolution unless that causes an OOM error.
Always set checkpoint saving to every epoch (e.g. --save_epoch_freq 1 for pix2pix-style repos)
so there is proof a checkpoint was written.
Monitor GPU memory with: watch -n 1 nvidia-smi (run in a second terminal).
-->

- **Script / command run:**
    
    ```bash
    # paste exact command here
    ```
    
- **Dataset used:**
- **Epochs run:**
- **Batch size:**
- **Input resolution:**
- **Time per epoch (approx):**
- **Peak GPU memory (approx, from nvidia-smi):**
- **Checkpoint saved:** yes / no
- **Checkpoint path:**
- **Crash or error during training:**
<!-- "None" if clean. Otherwise paste the key error line and the fix applied. -->

---

## Output Verification

<!--
Open 3-5 output images and compare them visually against the ground-truth target.
This is not a metric -- just a check that the model produced something in the right domain.
"Expected domain" for BCI would be IHC HER2-stained tissue with brown DAB staining on a light background.
Record one or two example output filenames so the check is reproducible.
-->

- **Output folder:**
- **Example output filenames:**
- **Dimensions match input:** yes / no
- **Visual sanity check:**
<!-- e.g. "outputs show IHC-like staining, structures roughly aligned with H&E input" -->
- **Any obvious artifacts or failure modes:**

---

## Changes Made to Original Code

<!--
Record every change made to the original repo, no matter how small.
Do not make changes that alter model architecture or training logic.
Only changes needed for the code to run in the benchmark environment are allowed.
Add rows as needed.
-->

| File | Change Description | Reason |
| --- | --- | --- |
|  |  |  |
|  |  |  |

<!--
Common examples of acceptable changes:

- Pinning a dependency version in requirements.txt (e.g. torch==2.1.0) because no version was specified
- Replacing a hardcoded absolute path with a command-line argument
- Removing an import that is not used and is not installable in the benchmark environment
- Adapting the data loader to accept BCI/MIST-HER2 folder structure
-->

---

## Frozen Environment

<!--
After the smoke test passes, export and commit the environment file.
Command: conda env export > environment_<model-name>.yml
This file is what gets adapted for the HPC migration later.
Note any packages that are unusual, very large, or likely to cause conflicts on the cluster.
-->

- **Environment file:** `environment_<model-name>.yml`
- **Committed to fork:** yes / no
- **Notes on unusual or heavy dependencies:**
<!-- e.g. "requires openslide-python which needs a system-level apt install" -->

---

## HPC Readiness Notes

<!--
Fill this in after the local smoke test passes.
Flag anything that will need attention before running on the VSC cluster.
Common issues: GUI/display dependencies (matplotlib backends), hardcoded CUDA package versions,
dependencies that require apt/system installs, very large model downloads.
Leave blank until local test is complete.
-->

- **Display/GUI dependencies to remove or neutralize:**
- **System-level dependencies (non-pip/conda):**
- **Estimated GPU memory requirement:**
- **Estimated storage requirement (weights + data):**
- **Other notes for cluster adaptation:**

---

## Summary

<!--
Write 2-4 sentences summarizing what worked, what did not, and what the next step is.
Be specific. Include the overall pass/fail verdict.
This is the first thing someone reads when picking this model back up.
-->

**Overall result:** PASS / FAIL / PARTIAL

<!-- Example pass:
"[Model] smoke test completed on [date]. Inference with pretrained weights passed on 10 BCI test images.
Training ran for 5 epochs without crash. One change was made to the data loader to accept separate
source/target folders. Frozen environment committed. Ready for full benchmark run."

Example fail:
"[Model] smoke test failed at the environment step. The required PyTorch version (1.4) is not
compatible with CUDA 12.1. Blocked until a workaround is identified. Do not schedule for HPC."
-->