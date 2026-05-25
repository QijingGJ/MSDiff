# Synthetic contrast-free LGE via diffusion-based framework in acute MI for image quality and quantitative scar analysis ([pdf]())

<img width="1191" height="909" alt="image" src="https://github.com/user-attachments/assets/8e1765dd-4c12-4292-9a43-7bea1bab72bd" />


## Training

python scripts/image_train.py --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 16


## Sampling

python scripts/image_sample.py --model_path ./result_model/20250505/model050000.pt --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False

## Thanks

Thanks to the base code [IDDPM](https://github.com/openai/improved-diffusion)

## Citation
Qi J, Yue X, Hu M, et al. Synthetic Contrast-Free LGE via Diffusion-Based Framework in Acute MI for Image Quality and Quantitative Scar Analysis. Circ Cardiovasc Imaging. 2026;19(2):e018967.
