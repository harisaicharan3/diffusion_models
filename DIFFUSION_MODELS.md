# 🧠 Diffusion Models Explained

This guide gives a concise, practical overview of how modern text-to-image diffusion models (like Stable Diffusion) work, focusing on the concepts you need to understand and operate them effectively.

## 1) Core Intuition
Diffusion models learn to turn noise into data. Training teaches a neural network to gradually remove noise from a noisy sample until a clean image emerges. At inference, you start from pure noise and iteratively denoise to synthesize an image that matches a prompt.

## 2) Forward (Noising) Process
During training, an image \(x_0\) is progressively noised over T steps producing \(x_t\):
- \(x_t = \sqrt{\alpha_t} \cdot x_0 + \sqrt{1 - \alpha_t} \cdot \epsilon\), where \(\epsilon \sim \mathcal{N}(0, I)\)
- \(\alpha_t\) is a schedule controlling how much noise is added per step.

The model is trained to predict the noise \(\epsilon\) given \(x_t\) (and conditioning like a text embedding).

## 3) Reverse (Denoising) Process
At inference, start with \(x_T \sim \mathcal{N}(0, I)\) and iteratively apply a learned denoiser to move from \(x_t\) to \(x_{t-1}\). After T steps, you obtain \(x_0\): the generated image.

In practice, we use a UNet to predict noise (or sometimes the data/velocity), and schedulers approximate the reverse process.

## 4) Classifier‑Free Guidance (CFG)
To align images with prompts, classifier‑free guidance runs the UNet twice per step:
- Conditional: conditioned on the prompt embedding
- Unconditional: with an “empty” prompt

Then it combines them:
- \(\hat{\epsilon} = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})\)
- Guidance scale \(s\) (a.k.a. guidance_scale) controls adherence to the prompt. Typical values: 6–9.

Trade‑offs:
- Low scale → more diverse but less on‑prompt
- High scale → more on‑prompt but risk of artifacts/saturation

## 5) Schedulers (Samplers)
Schedulers define how you step through noise levels and combine predictions. Common ones:
- DDIM: fast, deterministic sampling
- Euler / Euler Ancestral: popular trade‑offs
- DPM‑Solver (Multistep): high quality with fewer steps

More steps generally improve fidelity but cost time. Good defaults: 20–30 steps with DPM‑Solver.

## 6) Model Components
- Text Encoder: usually CLIP text encoder → turns prompt into embeddings
- UNet: predicts noise at each step, conditioned on text + timestep
- VAE: Stable Diffusion operates in latent space; VAE encodes/decodes between pixel and latent space
- Safety Checker (optional): filters NSFW content

## 7) Key Inference Knobs
- steps (num_inference_steps): 20–30 good baseline
- guidance_scale: 6–9 typical
- width/height: 512×512 default; larger needs more VRAM
- seed: set for reproducibility; vary for diversity
- negative_prompt: steer away from artifacts (e.g., “blurry, low quality, distorted”)

## 8) Memory & Performance
- Larger resolutions and models (e.g., SDXL) need more VRAM
- Enable memory‑efficient attention / CPU offloading if tight on VRAM
- Use smaller sizes or fewer steps for faster, lighter runs

## 9) Typical Failure Modes
- Washed‑out or over‑sharpened images → extreme guidance_scale
- Repeated objects / weird artifacts → tweak negative_prompt; try different seeds
- Low detail → increase steps; try a different scheduler

## 10) What Our App Does
- Loads a diffusion pipeline (UNet + VAE + text encoder + scheduler)
- Builds generation kwargs (prompt, steps, guidance_scale, width/height, etc.)
- Calls the pipeline (it defines `__call__`) to run iterative denoising
- Returns PIL images for preview, processing, and export

## 11) Further Reading
- Diffusers docs: `https://huggingface.co/docs/diffusers`
- Stable Diffusion paper: “High‑Resolution Image Synthesis with Latent Diffusion Models”
- Classifier‑Free Guidance: `https://arxiv.org/abs/2207.12598`

---
If you want a deeper dive (training loss, variance schedules, parameterization choices like \(\epsilon\)/\(v\)/\(x_0\)), we can add an advanced appendix as well.
