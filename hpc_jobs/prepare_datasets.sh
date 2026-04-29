#!/bin/bash
# prepare_datasets.sh
# Run once manually on the login node after datasets have been transferred via rsync.
# Creates symlinked trainA/trainB/testA/testB folders on scratch for CycleGAN/pix2pix.
# The model expects separate A and B folders -- no combine_A_and_B.py step is needed.
#
# Expected dataset layout after rsync:
#   $VSC_SCRATCH/dataset/BCI/
#     HE/train/   HE/test/
#     IHC/train/  IHC/test/
#
#   $VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/
#     trainA/  trainB/  valA/  valB/
#
# Usage:
#   bash prepare_datasets.sh
#
# Do NOT submit this with sbatch.
set -euo pipefail

echo "=== Verifying source dataset folders ==="
for DIR in \
    "$VSC_SCRATCH/dataset/BCI/HE/train" \
    "$VSC_SCRATCH/dataset/BCI/HE/test" \
    "$VSC_SCRATCH/dataset/BCI/IHC/train" \
    "$VSC_SCRATCH/dataset/BCI/IHC/test" \
    "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/trainA" \
    "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/trainB" \
    "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/valA" \
    "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/valB"
do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: Missing folder: $DIR"
        echo "Run the rsync transfer from your local machine first."
        exit 1
    fi
    echo "  OK: $DIR"
done

echo ""
echo "=== Cleaning up Zone.Identifier files from Windows transfer ==="
find "$VSC_SCRATCH/dataset/BCI" -name "*.Identifier" -delete && echo "  BCI cleaned"
find "$VSC_SCRATCH/dataset/MIST/HER2-004" -name "*.Identifier" -delete && echo "  MIST-HER2 cleaned"

echo ""
echo "=== Creating CycleGAN/pix2pix symlink folders on scratch ==="

# BCI
mkdir -p "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-BCI"
ln -sfn "$VSC_SCRATCH/dataset/BCI/HE/train"  "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-BCI/trainA"
ln -sfn "$VSC_SCRATCH/dataset/BCI/IHC/train" "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-BCI/trainB"
ln -sfn "$VSC_SCRATCH/dataset/BCI/HE/test"   "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-BCI/testA"
ln -sfn "$VSC_SCRATCH/dataset/BCI/IHC/test"  "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-BCI/testB"
echo "  pytorch-cyclegan-pix2pix-BCI symlinks created"

# MIST-HER2 (val split is used as test)
mkdir -p "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB"
ln -sfn "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/trainA" "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/trainA"
ln -sfn "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/trainB" "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/trainB"
ln -sfn "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/valA"   "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/testA"
ln -sfn "$VSC_SCRATCH/dataset/MIST/HER2-004/TrainValAB/valB"   "$VSC_SCRATCH/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/testB"
echo "  pytorch-cyclegan-pix2pix-MIST-HER2 symlinks created"

echo ""
echo "=== Image counts ==="
echo "  pytorch-cyclegan-pix2pix-BCI/trainA : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-BCI/trainA -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-BCI/trainB : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-BCI/trainB -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-BCI/testA  : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-BCI/testA  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-BCI/testB  : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-BCI/testB  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-MIST-HER2/trainA : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/trainA -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-MIST-HER2/trainB : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/trainB -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-MIST-HER2/testA  : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/testA  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  pytorch-cyclegan-pix2pix-MIST-HER2/testB  : $(find "$VSC_SCRATCH"/pytorch-cyclegan-pix2pix-MIST-HER2/TrainValAB/testB  -maxdepth 1 \( -type f -o -type l \) | wc -l)"

echo ""
echo "Done. Next step: sbatch install.sh"