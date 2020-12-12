# SinGAN for Image Inpainting

This repo contains the SinGAN repository with slightly modified files to work for Image Inpainting.

To run training on an image (make sure you have a GPU), you will first need to downgrade PyTorch:

```pip install torch==1.4.0 torchvision==0.5.0```

Then run the following line:

```python main_train.py --input_name [YOUR IMAGE NAME].png```


To perform inpainting, after having trained on your image, run:

```!python inpainting.py --input_name <YOUR IMAGE NAME>.png --ref_name <YOUR EDIT NAME>.png --inpainting_start_scale <CHOOSE SCALE>```

You will need to provide in `Input/Inpainting/` the edited image (i.e. with masked holes) and the mask (`<YOUR EDIT NAME>_mask.png`) which is a black and white image.


## TODO

- Implement PSNR for comparison for small holes
- Rewrite SinGAN model to use partial convolutions and a mask
- Implement different "masking" or "editing" strategies on the image to train: either using the mean of the rest of the image, or other

