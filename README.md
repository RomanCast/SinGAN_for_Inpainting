# SinGAN for Image Inpainting

This repo is implementing SinGAN for Image Inpainting. It builds mostly on the [original SinGAN repository](https://github.com/tamarott/SinGAN), implements different initialisation strategies and contains an implementation of SinGAN with Partial Convolutions, with code to downsample masks for each scale.

To run training on an image (make sure you have a GPU), you will first need to downgrade PyTorch:

```
pip install torch==1.4.0 torchvision==0.5.0
```

## Run SinGAN with Partial Convolutions
Run the following line to train on an image using Partial Convolutions. You will need the occluded image, the same image in the `Input/Inpainting` folder, as well as a mask in the same folder `Input/Inpainting/<IMAGE_NAME>_edit_mask.png`. The mask has to be a binary image.

```
python main_train_partial.py --input_name <IMAGE_NAME>.png --ref_name <IMAGE_NAME>_edit.png
```

## Initialisation strategies when inpainting
To perform inpainting after having trained a model either with SinGAN or SinGAN with Partial Convolutions, run the following line. The flag `--partial` specifies which model to use. The `--fill` option can be one of the following:

- `None` : leaves the hole blank
- `mean` : fills with average pixel value in the non-occluded region
- `localMean` : fills with average pixel value in a small zone around the hole
- `NNs` : simple version of the PatchMatch algorithm that finds nearest neighbours in the image

```
python inpainting.py --input_name <IMAGE_NAME>.png --ref_name <IMAGE_NAME>_edit.png --partial --inpainting_start_scale 2 --fill <FILL>
```

This repo relies on Partial Convolutions and SinGAN, cite them if needed:

```
@inproceedings{shaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Shaham, Tamar Rott and Dekel, Tali and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4570--4580},
  year={2019}
}

@inproceedings{liu2018image,
  title={Image Inpainting for Irregular Holes using Partial Convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```
