import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085,
                 beta_end: float = 0.0120):
        """
        Implements sampling operations in a Denoising Diffusion Probabilistic Model (DDPM), facilitating the controlled
        transformation of noise into coherent images or data over a series of steps.

        :param generator: A torch.Generator object for random number generation, ensuring reproducibility.
        :param num_training_steps: Total number of training timesteps used in the diffusion process.
        :param beta_start: Initial value of the variance schedule's beta, setting the starting noise level.
        :param beta_end: Final value of the variance schedule's beta, setting the ending noise level.
        """

        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)

        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

        self.start_step = None
        self.num_inference_steps = None

    def set_inference_timesteps(self, num_inference_steps=50):
        """
        Adjusts the number of inference timesteps based on the desired resolution of the sampling process.

        :param num_inference_steps: The desired number of inference steps to be used during generation.
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Calculates the previous timestep index based on the current timestep.

        :param timestep: Current timestep index.
        :return: Previous timestep index.
        """

        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps

        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Calculates the variance for the noise model at a given timestep.

        :param timestep: Index of the current timestep.
        :return: Variance tensor for the current timestep.
        """

        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image.
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.

        :param strength: Value between 0 and 1 indicating the degree to which the noise should influence the final image.
                         A strength near 1 indicates maximum noise and regeneration, while near 0 indicates minimal noise.
        """

        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Performs a single step of the reverse diffusion process, predicting the previous timestep's latent state.

        :param timestep: Current timestep index.
        :param latents: Current latent variables tensor.
        :param model_output: Model's prediction for the current timestep.
        :return: Updated latents for the previous timestep.
        """

        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise

        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Applies controlled noise to the original samples based on the diffusion schedule for specified timesteps,
        following the q(x_t | x_0) model of adding Gaussian noise. This simulates the forward process of the diffusion,
        preparing samples for subsequent reverse diffusion denoising steps.

        :param original_samples: Tensor of original data samples, shape (batch_size, channels, height, width).
        :param timesteps: Indices of the diffusion timesteps to determine the noise level for each sample.
        :return: Tensor of noisy samples, same shape as `original_samples`.
        """

        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device,
                            dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples
