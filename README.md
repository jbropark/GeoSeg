
## Introduction

This is a GeoSeg lightweight project. Please visit the [original repository](https://github.com/WangLibo1995/GeoSeg). 

**GeoSeg** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on developing advanced Vision Transformers for remote sensing image segmentation.


## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── GeoSeg (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
    ├── LoveDA
        ├── Train
        │   ├── Urban
        │   │   ├── images_png (original images)
        │   │   ├── masks_png (original masks)
        │   │   ├── masks_png_convert (converted masks used for training)
        │   │   ├── masks_png_convert_rgb (original rgb format masks)
        │   ├── Rural
        │       ├── images_png 
        │       ├── masks_png 
        │       ├── masks_png_convert
        │       ├── masks_png_convert_rgb
        ├── Val (the same with Train)
        ├── Test
        ├── train_val (Merge Train and Val)

```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**LoveDA**
```
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

## Training

We have three training process
- teacher net
- student net
- prune net

For each process, you should provide proper config file to the training script.

"-c" means the path of the config, use different **config** to train different models.

```
python GeoSeg/train_supervision.py -c GeoSeg/config/loveda/unetformer_teacher.py
python GeoSeg/train_supervision_distill.py -c GeoSeg/config/loveda/unetformer_distill.py
python GeoSeg/train_supervision.py -c GeoSeg/config/loveda/unetformer_prune.py
```

## Prune

You can prune the student net. Pruner uses [Torch-Pruning](https://github.com/VainF/Torch-Pruning).

```
python GeoSeg/loveda_prune.py \
  --ratio=0.3 \
  --importance=mag \
  --path="model_weights/loveda/resnet-distill/resnet-distill.ckpt" \
  --output="prune.ckpt"
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"--rgb" denotes whether to output masks in RGB format

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/unetformer.py -o fig_results/loveda/test
```

## Citation

If you find this project useful in your research, please consider citing：

- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)
- [A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/9681903) 
- [Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images](https://www.mdpi.com/2072-4292/13/16/3065)
- [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
- [Multiattention network for semantic segmentation of fine-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/9487010)
- [A2-FPN for semantic segmentation of fine-resolution remotely sensed images](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)


## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
