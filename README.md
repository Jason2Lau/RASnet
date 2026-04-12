# NTIRE 2026 Challenge on Image Shadow Removal

<div align=center>
<img src="https://github.com/Jason2Lau/RASnet/blob/master/figs/logo.png" width="400px"/> 
</div>

RASnet borrows the stage 1 and stage 3 training processes from the paper [Dereflection Any Image with Diffusion Priors and Diversified Data](https://arxiv.org/pdf/2503.17347), and extends them to become the algorithm used in the NTIRE 2026 Image Shadow Removal Challenge.
We make the following modifications to the original method:
- 1. In the stage 1, a pixel-space loss function is introduced to alleviate the problem of blurry generated images, while the features of dinov3 are used to enhance shadow removal capabilities.
- 2. We skipped stage 2 because we found it had limited impact on the final result in this task.
- 3. In the stage 3, we introduce FFL (Focal Frequency Loss) to help VAE better reconstruct image details.
## QUICK START
- Environment
    - pip3 install -r requirements.txt
    - Our experiment was run on CUDA 11.8， and the same configuration is recommended

- Pre-trained Models
    - Ensure that the model weights are stored in the same directory as below.

    | dirname                                          | download link | dir withing the code |
    |--------------------------------------------------|---------------|----------------------|
    | sd                                               |        https://drive.google.com/drive/folders/1w71cca44VKyhMZHX0lqCNDJKkpl1B_ha?usp=drive_link       | weights/sd           |
    | controlnet                                       |   https://drive.google.com/drive/folders/1H6xlZ30U5PTrLXlStO9TuNy6SykBPB8O?usp=drive_link            | weights/controlnet   |
    | cross_vae                                        |        https://drive.google.com/drive/folders/1Bk2TSmPLWDACSY4If_DTGXbIPo2iS50g?usp=drive_link       | weights/cross_vae    |

- Training
    - code will be released soon
  
- Inference
    - example
        ```
        python3 test.py --pretrained_dai weights/sd --controlnet weights/unet/checkpoint.pt  --cross_vae weights/cross_vae --input_dir inputs/ntire26_shadow_test_in --output_dir outputs/results/ntire26_shadow_test_out
        ```
- Results
  - You can download our results via the following link:
    - https://drive.google.com/file/d/1CGZq0mwFpmISa7Kf1lipWFBQDnAimM_F/view?usp=drive_link