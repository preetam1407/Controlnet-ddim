import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ######## TODO ########
        # Compute xt.
        # Simulate the forward diffusion process by adding noise to clean data.
        # Calculate xt = sqrt(alphā_t) * x0 + sqrt(1 - alphā_t) * noise
        # This gives a noisy version of the input at timestep t.
        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        sqrt_alphas_prod_t = alphas_prod_t.sqrt()
        sqrt_one_minus_alphas_prod_t = (1 - alphas_prod_t).sqrt()
        xt = sqrt_alphas_prod_t * x0 + sqrt_one_minus_alphas_prod_t * noise
        #######################

        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})

        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        # Reverse one step in the diffusion process: predict x_{t-1} from x_t.
        # First, predict the noise with the network.
        # Then, calculate the mean and add noise based on beta_t to get x_{t-1}.
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
        ).sqrt()
        eps_theta = self.network(xt, t)

        alpha_t = extract(self.var_scheduler.alphas, t, xt)
        coef = 1.0/alpha_t.sqrt()
        mean = coef * (xt - eps_theta * eps_factor)
        noise = torch.randn_like(xt)
        beta_t = extract(self.var_scheduler.betas, t, xt)
        sigma = beta_t.sqrt().view(-1, *([1] * (xt.ndim - 1)))
        mask = (t > 0).float().view(-1, *([1] * (xt.ndim - 1)))
        x_t_prev = mean + mask * sigma * noise

        #######################
        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # Start from random noise and progressively denoise it step by step.
        # Loop backward from timestep T-1 to 0, applying p_sample at each step.
        # sample x0 based on Algorithm 2 of DDPM paper.
        xt = torch.randn(shape, device=self.device)

        # 2) Walk through timesteps [T-1, T-2, ..., 0]
        for t in self.var_scheduler.timesteps:
            # build a batch‑sized tensor of the current timestep
            batch_size = shape[0]
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            # denoise one step
            xt = self.p_sample(xt, t_batch)

        # 3) After the loop, xt is our final denoised sample
        x0_pred = xt
        return x0_pred

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        One step denoising function of DDIM: x_{τ_i} -> x_{τ_{i-1}}.
        """
        # 1) get ᾱ for current and previous timesteps
        alpha_prod_t      = extract(self.var_scheduler.alphas_cumprod, t, xt)      # ᾱ_{τ_i}
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)  # ᾱ_{τ_{i-1}}
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        # 2) predict noise
        eps_theta = self.network(xt, t)

        # 3) estimate x0
        #    x0_pred = (xt - √(1−ᾱ_t)·εθ) / √ᾱ_t
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_1_alpha_prod = (1 - alpha_prod_t).sqrt()
        x0_pred = (xt - sqrt_1_alpha_prod * eps_theta) / sqrt_alpha_prod_t

        # 4) compute sigma for the DDIM update
        #    σ = η · sqrt((1−ᾱ_prev)/(1−ᾱ_t)) · sqrt(1−ᾱ_t/ᾱ_prev)
        sigma = eta * torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) \
                      * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev)

        # 5) compute the “non‑stochastic” part coefficient
        #    c = √(1−ᾱ_prev − σ²)
        c = torch.sqrt(torch.clamp(1 - alpha_prod_t_prev - sigma**2, min=0.0))

        # 6) sample noise
        noise = torch.randn_like(xt)

        # 7) combine to get x_{τ_{i-1}}
        x_t_prev = sqrt_alpha_prod_t_prev.unsqueeze(-1) * x0_pred \
                   + c.unsqueeze(-1) * eps_theta \
                   + sigma.unsqueeze(-1) * noise

        return x_t_prev


    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
        """
        The loop of the reverse process of DDIM.
        """
        # compute which of the original T timesteps we’ll actually use
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps).to(self.device)
        prev_timesteps = timesteps - step_ratio

        # start from pure noise
        xt = torch.randn(shape, device=self.device)

        # reverse through the “coarse” DDIM steps
        batch_size = shape[0]
        for t, t_prev in zip(timesteps, prev_timesteps):
            # build batch-sized tensors for current and next timesteps
            t_batch      = torch.full((batch_size,),     t,      device=self.device, dtype=torch.long)
            t_prev_batch = torch.full((batch_size,),     t_prev, device=self.device, dtype=torch.long)
            # one DDIM denoising step
            xt = self.ddim_p_sample(xt, t_batch, t_prev_batch, eta)

        # after all steps, xt is our estimate of x0
        x0_pred = xt
        return x0_pred


    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.

        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        # 1) batch size and random timesteps t ~ Uniform({0,…,T−1})
        batch_size = x0.shape[0]
        t = (
            torch.randint(
                0,
                self.var_scheduler.num_train_timesteps,
                size=(batch_size,),
            )
            .to(x0.device)
            .long()
        )

        # 2) sample noise ε ~ N(0,I)
        noise = torch.randn_like(x0)

        # 3) form the noisy inputs x_t via the forward process
        xt = self.q_sample(x0, t, noise)

        # 4) let the network predict the noise from (x_t, t)
        eps_theta = self.network(xt, t)

        # 5) mean‑squared error between true noise and predicted noise
        loss = F.mse_loss(eps_theta, noise)

        return loss


    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
