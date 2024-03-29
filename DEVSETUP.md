My Dev Setup Notes

on HPC, export TMPDIR=/home/n11020211/tmp  so that pdm works

Grab training data into ~/datasets/LLaVA-Pretrain and ~/datasets/LLaVA-InstructionTune

$ mamba env create -f environment.yaml
$ conda activate llava
$ pdm install

# setup data
cd playground/data
ln -s ~/datasets/LLaVA-Pretrain LLaVA-Pretrain

ln -s ~/datasets/LLaVA-InstructionTune/coco coco

ln -s ~/datasets/LLaVA-InstructionTune/gqa gqa
ln -s ~/datasets/LLaVA-InstructionTune/ocr_vqa ocr_vqa
ln -s ~/datasets/LLaVA-InstructionTune/textvqa textvqa
ln -s ~/datasets/LLaVA-InstructionTune/vg vg

# train

bash ./scripts/v1_5/


