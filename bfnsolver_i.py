
import torch
import os
from omegaconf import OmegaConf, DictConfig
from utils import seed_everything, make_config, get_nnet
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class ImageBFNSolver:
    def __init__(self, unet: torch.nn.Module, num_steps: int = 100, min_variance: float = 1e-6, eta: float = 1e-5):
        
        self.unet = unet
        self.sigma_1 = min_variance**0.5

        self.num_steps = num_steps
        self.steps = torch.flip(torch.arange(num_steps+1), [0])
        self.times = self.steps.to(torch.float64)/(num_steps) * (1 - eta)
        self.delta_t = (1 - eta) / num_steps

    
        self.beta_s  = self.sigma_1**(-2 * (1 - self.times)) - 1
        self.gamma_t = 1 - self.sigma_1**(2 * (1 - self.times))
        self.alpha_t = 1 - self.sigma_1**(2 * (1 - self.times))
        self.sigma_t = (self.alpha_t * (1 - self.alpha_t)).sqrt()
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        self.f_t = 2 * torch.log(torch.tensor(self.sigma_1)) * (1 - self.gamma_t) / self.gamma_t
        self.g_t = (-2 * torch.log(torch.tensor(self.sigma_1)) * (1 - self.gamma_t))**0.5

    def sde_euler_update(self, x_s, step, last_drop=False):
        # x_s -> x_t
        t  = torch.ones_like(x_s, device=x_s.device) * (1 - self.times[step])
        # noise predict and x0 predict
        with torch.no_grad():
            noise_pred = self.unet(x_s, t).reshape(x_s.shape)
        alpha_t, sigma_t = self.alpha_t[step], self.sigma_t[step]
        x0_pred = (x_s - sigma_t * noise_pred) / alpha_t
        
        # clip x0
        x0_pred = x0_pred.clip(min=-1.0, max=1.0)
        noise_pred = (x_s - x0_pred * alpha_t) / sigma_t

        beta_t, beta_s = self.beta_s[step + 1], self.beta_s[step]
        gamma_s = self.gamma_t[step]
        f = self.f_t[step]
        g = self.g_t[step]
        noise = torch.randn_like(x_s, device=x_s.device)

        if last_drop == True and step == self.num_steps - 1:
            return x0_pred, x0_pred
        else:
            # bfn
            # x_t = ((beta_t - beta_s) / ((beta_t + 1) * gamma_s) + (beta_s + 1)/(beta_t + 1)) * x_s - (beta_t - beta_s) / (beta_t + 1) * ((1-gamma_s)/gamma_s)**0.5 * noise_pred + (beta_t-beta_s)**0.5/(beta_t + 1) * noise
            # sde
            x_t = x_s - (f * x_s + g**2 * noise_pred / (gamma_s * (1-gamma_s))**0.5) * self.delta_t + g * self.delta_t**0.5 * noise
        return x_t, x0_pred
        
    def ode_bfnsolver1_update(self, x_s, step, last_drop=False):
        # x_s -> x_t
        t = torch.ones_like(x_s, device=x_s.device) * (1 - self.times[step])
        # noise predict and x0 predict
        with torch.no_grad():
            noise_pred = self.unet(x_s, t).reshape(x_s.shape)
        alpha_t, sigma_t = self.alpha_t[step], self.sigma_t[step]
        x0_pred = (x_s - sigma_t * noise_pred) / alpha_t

        # clip x0
        x0_pred = x0_pred.clip(min=-1.0, max=1.0)
        noise_pred = (x_s - x0_pred * alpha_t) / sigma_t

        # get schedule
        lambda_t, lambda_s = self.lambda_t[step + 1], self.lambda_t[step]
        alpha_t, alpha_s = self.alpha_t[step + 1], self.alpha_t[step]
        sigma_t, sigma_s = self.sigma_t[step + 1], self.sigma_t[step]
        h = lambda_t - lambda_s

        if last_drop == True and step == self.num_steps - 1:
            return x0_pred, x0_pred
        else:
            x_t = (alpha_t / alpha_s) * x_s - (sigma_t * (torch.exp(h) - 1.0)) * noise_pred

        return x_t, x0_pred

    def ode_bfnsolver2_multi_step_update(self, x_s0, step, x0_pred_last=None, last_drop=False):
        lambda_t, lambda_s0 = self.lambda_t[step + 1], self.lambda_t[step],
        alpha_t, alpha_s0 = self.alpha_t[step + 1], self.alpha_t[step]
        sigma_t, sigma_s0 = self.sigma_t[step + 1], self.sigma_t[step]
        h = lambda_t - lambda_s0

        timestep_s0 = torch.ones_like(x_s0, device=x_s0.device) * (1 - self.times[step])
        with torch.no_grad():
            noise_pred_s0 = self.unet(x_s0, timestep_s0).reshape(x_s0.shape)
        x0_pred_s0 = (x_s0 - sigma_s0 * noise_pred_s0) / alpha_s0
        x0_pred_s0 = x0_pred_s0.clip(-1, 1)

        if step == 0:
            x_t = (sigma_t / sigma_s0) * x_s0 - alpha_t * (torch.exp(-h) - 1) * x0_pred_s0
            return x_t, x0_pred_s0
        elif last_drop == True and step == self.num_steps - 1:
            return x0_pred_s0, x0_pred_s0
        else:
            lambda_s1 = self.lambda_t[step - 1]
            h_0 = lambda_s0 - lambda_s1
            r = h_0 / h

            D = (1 + 1 / (2 * r)) * x0_pred_s0 - 1 / (2 * r) * x0_pred_last
            x_t = sigma_t / sigma_s0 * x_s0 - alpha_t * (torch.exp(-h) - 1) * D
            return x_t, x0_pred_s0

    def sde_bfnsolver2_multi_step_update(self, x_s0, step, x0_pred_last=None, last_drop=False):
        lambda_t, lambda_s0 = self.lambda_t[step + 1], self.lambda_t[step],
        alpha_t, alpha_s0 = self.alpha_t[step + 1], self.alpha_t[step]
        sigma_t, sigma_s0 = self.sigma_t[step + 1], self.sigma_t[step]
        h = lambda_t - lambda_s0

        timestep_s0 = torch.ones_like(x_s0, device=x_s0.device) * (1 - self.times[step])
        with torch.no_grad():
            noise_pred_s0 = self.unet(x_s0, timestep_s0).reshape(x_s0.shape)
        x0_pred_s0 = (x_s0 - sigma_s0 * noise_pred_s0) / alpha_s0
        x0_pred_s0 = x0_pred_s0.clip(-1, 1)

        noise = torch.randn_like(x_s0, device=x_s0.device)
        if step == 0:
            x_t = (sigma_t / sigma_s0 * torch.exp(-h)) * x_s0 + \
                  alpha_t * (1 - torch.exp(-2.0 * h)) * x0_pred_s0 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2 * h)) * noise
            return x_t, x0_pred_s0
        elif last_drop == True and step == self.num_steps - 1:
            return x0_pred_s0, x0_pred_s0
        else:
            lambda_s1 = self.lambda_t[step - 1]
            h_0 = lambda_s0 - lambda_s1  # 和paper相反，和diffusers一致
            r = h_0 / h
            D1 = (x0_pred_s0 - x0_pred_last) / r
            x_t = (sigma_t / sigma_s0 * torch.exp(-h)) * x_s0 + alpha_t * (1 - torch.exp(-2.0 * h)) * x0_pred_s0 \
                  + 0.5 * alpha_t * (1 - torch.exp(-2.0 * h)) * D1 + sigma_t * torch.sqrt(
                1.0 - torch.exp(-2.0 * h)) * noise
            return x_t, x0_pred_s0

@torch.inference_mode()
def sample(cfg: DictConfig):
    train_cfg = make_config(cfg.config_file)
    unet = get_nnet(**train_cfg.net)

    if torch.cuda.is_available():
        unet.cuda()

    cfg = OmegaConf.merge(train_cfg.sampling, cfg)
    state_dict = torch.load(cfg.load_model, map_location='cpu')
    new_state_dict = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    unet.load_state_dict(new_state_dict)
    unet.eval()
    logging.info(f"Loaded model from {cfg.load_model}") 
    

    seed_everything(cfg.seed)
    logging.info(f"Seeded everything with seed {cfg.seed}")
    
    logging.info(f"sample with {cfg.algorithm}")
    logging.info(f"number of samples: {cfg.batch_size}, n_steps: {cfg.n_steps}, last_drop: {cfg.last_drop}, eta: {cfg.eta}, initial_dist: {cfg.initial_dist}")

    samples_shape = (cfg.batch_size, train_cfg.data.image_size, train_cfg.data.image_size, train_cfg.data.image_channels)
    min_variance = train_cfg.bayesian_flow.min_variance
    # BFN solvers
    bfnsolver = ImageBFNSolver(unet, num_steps=cfg.n_steps, min_variance=min_variance, eta=cfg.eta)
    
    samples = batch_sample(cfg, train_cfg, unet, bfnsolver, samples_shape, min_variance)
    
    cfg.save_path = f"./samples/cifar/{cfg.algorithm}/n_steps={cfg.n_steps}"
    logging.info(f"save samples to {os.path.dirname(cfg.save_path)}")
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    save_batch(samples, cfg.save_path)


def batch_sample(cfg: DictConfig, train_cfg: DictConfig, unet: torch.nn.Module, bfnsolver: ImageBFNSolver, samples_shape: tuple, min_variance: float):
    # Choices of the Initialization Distribution
    if cfg.initial_dist == "zero_mean_and_std":
        gamma_t = 1 - min_variance**cfg.eta
        std_t = (gamma_t * (1 - gamma_t))**0.5
        xt = torch.randn(*samples_shape).to(next(unet.parameters()).device) * std_t
    elif cfg.initial_dist == "optimal_mean_and_std":
        saved_tensors = torch.load(cfg.mean_std_path)
        mean_x = saved_tensors['mean'].to(next(unet.parameters()).device)
        std_x = saved_tensors['std'].to(next(unet.parameters()).device)
        trace_x = std_x.square().sum()
        gamma_t = 1 - min_variance**cfg.eta
        mean_t = gamma_t * mean_x
        std_t =  (gamma_t * (1 - gamma_t) + gamma_t**2 * trace_x / (train_cfg.data.image_size * train_cfg.data.image_size * train_cfg.data.image_channels))**0.5
        xt = torch.randn(*samples_shape).to(next(unet.parameters()).device) * std_t + mean_t

    data_pred_last = None
    for step in tqdm(range(cfg.n_steps)):
        if cfg.algorithm == "ode_euler":
            pass
        if cfg.algorithm == "sde_euler":
            xt, _ = bfnsolver.sde_euler_update(xt, step, last_drop=cfg.last_drop)
        elif cfg.algorithm == "ode_bfnsolver1":
            xt, _ = bfnsolver.ode_bfnsolver1_update(xt, step, last_drop=cfg.last_drop)
        elif cfg.algorithm == 'ode_bfnsolver2_multi_step':
            xt, data_pred_last = bfnsolver.ode_bfnsolver2_multi_step_update(xt, step, x0_pred_last=data_pred_last, last_drop=cfg.last_drop)
        elif cfg.algorithm == 'sde_bfnsolver2_multi_step':
            xt, data_pred_last = bfnsolver.sde_bfnsolver2_multi_step_update(xt, step, x0_pred_last=data_pred_last, last_drop=cfg.last_drop)
        else:
            raise NotImplementedError(cfg.algorithm)

    samples = xt
    return samples

def save_batch(samples: torch.Tensor, save_path: str) -> None:
        
        import numpy as np
        from PIL import Image
        samples = (samples.float() + 1) / 2 * 255
        # samples = ((samples.float() + 1) / 2 ).clamp(0, 1)* 255
        np_images = samples.cpu().detach().numpy().astype(np.uint8)
        if np_images.shape[0] <= 10:
            new_image = Image.fromarray(np.concatenate(np_images, axis=1))
        else:
            img_size = np_images.shape[1]
            new_image = Image.new('RGB', (10*img_size, np_images.shape[0] // 10*img_size))
            locations = [(i % 10 * img_size, i // 10 * img_size) for i in range(np_images.shape[0])]
            for (x, y), np_image in zip(locations, np_images):
                img = Image.fromarray(np_image)
                new_image.paste(img, (x, y))
        new_image.save(f"{save_path}.png")

if __name__ == "__main__":
    config = OmegaConf.from_cli()
    sample(config)