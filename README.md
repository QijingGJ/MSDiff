# Synthetic contrast-free LGE via diffusion-based framework in acute MI for image quality and quantitative scar analysis ([PDF](https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.125.018967?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed))

<img width="1065" height="1064" alt="image" src="https://github.com/user-attachments/assets/1330dc7f-67a8-4156-9ba2-b562830717b7" />



## Training

```python
python scripts/image_train.py --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 16
```

## Sampling

```python
python scripts/image_sample.py --model_path ./result_model/20250505/model050000.pt --image_size 128 --learn_sigma True --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
```

## Thanks

Thanks to the base code [IDDPM](https://github.com/openai/improved-diffusion)

## Citation

```python
Qi J, Yue X, Hu M, et al. Synthetic Contrast-Free LGE via Diffusion-Based Framework in Acute MI for Image Quality and Quantitative Scar Analysis. Circ Cardiovasc Imaging. 2026;19(2):e018967.
```
