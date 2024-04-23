import os
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical as TorchCategorical
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import logging

from utils import seed_everything, make_config, get_nnet, batch_to_str

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set global seeds and flags for reproducibility and performance
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
# torch.set_default_dtype(torch.float64)

class TextBFNSolver:
    def __init__(self, unet: torch.nn.Module, class_num: int = 27,
                 num_steps: int = 100, max_sqrt_beta: float = 0.75, eta: float = 1e-5):
        self.unet = unet
        
        self.max_sqrt_beta = max_sqrt_beta
        self.K = class_num

        self.num_steps = num_steps
        self.steps = torch.flip(torch.arange(num_steps+1), [0])
        self.times = self.steps.to(torch.float64) / num_steps *  (1 - eta)
        self.delta_t = (1 - eta) / num_steps
        
        
        # f g
        self.f_t = -2 / (1 - self.times)
        self.g_t = (2 * self.K * (1 - self.times))**0.5 * self.max_sqrt_beta

        # beta alpha
        self.beta_t  = (self.max_sqrt_beta * (1 - self.times))**2
        self.alpha_t = 2 * (1 - self.times) * self.max_sqrt_beta**2

    
    def sde_euler_update(self, x_s, step, last_drop=False, cate_samp=False, addi_step=False):
        # x_s -> x_t
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])

        g = self.g_t[step]

        noise = torch.randn_like(x_s, device=x_s.device)

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                categorical = TorchCategorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.K)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred    
            elif addi_step == True and step == self.num_steps - 1:
                x_t = x_s + g**2 * (data_pred - 1/self.K) * self.delta_t + g * self.delta_t**0.5 * noise
                theta = F.softmax(x_t, -1)
                t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step+1])
                logits = self.unet(theta, t)
                data_pred = F.softmax(logits, -1)
                return logits, data_pred
            else:
                x_t = x_s + g**2 * (data_pred - 1/self.K) * self.delta_t + g * self.delta_t**0.5 * noise
                return logits, data_pred

    def ode_euler_update(self, x_s, step, last_drop=False, cate_samp=False, addi_step=False):
        # x_s -> x_t
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])

        f = self.f_t[step]
        g = self.g_t[step]
        beta_s = self.beta_t[step]


        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                categorical = TorchCategorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.K)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            elif addi_step == True and step == self.num_steps - 1:
                x_t = x_s - ((f + (g**2)/(2 * self.K * beta_s)) * x_s - 0.5 * g**2 *(data_pred -1/self.K)) * self.delta_t
                theta = F.softmax(x_t, -1)
                t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step+1])
                logits = self.unet(theta, t)
                data_pred = F.softmax(logits, -1)
                return logits, data_pred
            else:
                x_t = x_s - ((f + (g**2)/(2 * self.K * beta_s)) * x_s - 0.5 * g**2 *(data_pred -1/self.K)) * self.delta_t
                return x_t, data_pred

    def ode_bfnsolver1_update(self, x_s, step, last_drop=False):
        # x_s -> x_t
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            else:
                x_t = (1-t_t)/(1-t_s) * x_s +c_t * (t_t -t_s) * ( 1 / self.K - data_pred)
                return x_t, data_pred
    
    def ode_bfnsolver2_multi_step_update(self, x_s, step, data_pred_last=None, last_drop=False):
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if step == 0:
                x_t = (1 - t_t) / (1 - t_s) * x_s + c_t * (t_t - t_s) * (1 / self.K - data_pred) 
                return x_t, data_pred
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            else:
                t_r = self.times[step - 1]
                # x_t = x_s + 
                A = (1 - t_t) / (1 - t_s) * x_s + c_t / self.K * (t_t - t_s)
                B = -c_t * (t_t - t_s) * data_pred
                D1 = (data_pred - data_pred_last)/(t_s - t_r)
                C = -c_t * (t_t - t_s)**2 / 2 * D1
                x_t = A + B + C
                return A + B + C, data_pred

    def ode_bfnsolver2_single_step_upate(self, x_s, step, last_drop=False):
        # x_s -> x_t
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        t_r = (t_t + t_s)/2
        c_r = self.K * self.max_sqrt_beta**2 * (1 - t_r)
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred_s = F.softmax(logits, -1)
        
            # x_r
            x_r = (1 - t_r)/(1 - t_s) * x_s + c_r * (t_r - t_s) * (1 / self.K - data_pred_s)
            t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - t_r)
            theta = F.softmax(x_r, -1)
            logits = self.unet(theta, t)
            data_pred_r = F.softmax(logits, -1)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred_r
            else:
                A = (1 - t_t)/ (1 - t_s) * x_s + c_t / self.K * (t_t - t_s)
                B = -c_t * (t_t - t_s) * data_pred_s
                D1 = (data_pred_r - data_pred_s)/(t_r - t_s)
                C = -c_t * (t_t - t_s)**2 / 2 * D1
                x_t = A + B + C
                return x_t, data_pred_r
    
    def sde_bfnsolver2_multi_step_update(self, x_s, step, data_pred_last=None, last_drop=False):
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s)**2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t)**2
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred_s = F.softmax(logits, -1)
            if step == 0:
                noise = torch.randn_like(x_s, device=x_s.device)
                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)  + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred_s
            else:
                noise = torch.randn_like(x_s, device=x_s.device)
                t_r = self.times[step-1]
                D1 = (data_pred_last - data_pred_s)/(t_r - t_s)
                # x_t_ = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)\
                #     + (2*self.K*self.max_sqrt_beta**2*( ((t_t**2)/2 - (t_t**3)/3) - ((t_s**2)/2-(t_s**3)/3 ) ) + t_s * self.K * (beta_t - beta_s)) * D1 \
                #         + (self.K * (beta_t - beta_s))**0.5 * noise

                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1) \
                    + 1/3 * self.K * self.max_sqrt_beta**2 * (t_t - t_s)**2 * (t_s + 2 * t_t -3) * D1 \
                    + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred_s

    def sde_bfnsolver1_update(self, x_s, step, last_drop=False, cate_samp=False):
        t = torch.ones(x_s.size()[:-1], device=x_s.device) * (1 - self.times[step])
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s)**2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t)**2
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            logits = self.unet(theta, t)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                data_pred = TorchCategorical(logits=logits, validate_args=False).sample()
                data_pred = F.one_hot(data_pred, self.K).to(torch.float32)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred
            else:
                noise = torch.randn_like(x_s, device=x_s.device)
                x_t = x_s + (beta_t - beta_s) * (self.K * data_pred - 1)  + (self.K * (beta_t - beta_s))**0.5 * noise
                return x_t, data_pred

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
    
    logging.info(f"Sample with {cfg.algorithm}")
    logging.info(f"Number of samples: {cfg.n_samples}, n_steps: {cfg.n_steps}, last_drop: {cfg.last_drop}, eta: {cfg.eta}, initial_dist: {cfg.initial_dist}, cate_samp: {cfg.cate_samp}")

    K = train_cfg.bayesian_flow.n_classes
    max_sqrt_beta = train_cfg.bayesian_flow.max_sqrt_beta

    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]
    
    idx = 0
    for batch_size in tqdm(amortize(cfg.n_samples, cfg.batch_size), desc='sample2dir'):
        samples_shape = (batch_size, train_cfg.data.seq_len)

        # BFN solvers
        bfnsolver = TextBFNSolver(unet, class_num=K, num_steps=cfg.n_steps, max_sqrt_beta=max_sqrt_beta, eta=cfg.eta)

        # Choices of the Initialization Distribution
        if cfg.initial_dist == "zero_mean_and_std":
            beta_t  = (max_sqrt_beta * cfg.eta)**2
            std_t = (K * beta_t)**0.5            
            xt = torch.randn(*samples_shape, K).to(next(unet.parameters()).device) * std_t
        elif cfg.initial_dist == "optimal_mean_and_std":
            saved_tensors = torch.load(cfg.mean_std_path)
            mean_ex = saved_tensors['mean'].to(next(unet.parameters()).device)
            std_ex = saved_tensors['std'].to(next(unet.parameters()).device)
            trace_ex = std_ex.square().sum()
            beta_t  = (max_sqrt_beta * cfg.eta)**2
            mean_t = beta_t * (K * mean_ex - 1)
            std_t = (K * beta_t  + K / train_cfg.data.seq_len * beta_t**2 * trace_ex)**0.5   
            xt = torch.randn(*samples_shape, K).to(next(unet.parameters()).device) * std_t + mean_t
        else:
            raise NotImplementedError(cfg.initial_dist)
        
        data_pred_last = None
        for step in tqdm(range(cfg.n_steps)):
            if cfg.algorithm == "sde_euler":
                xt, _ = bfnsolver.sde_euler_update(xt, step, cate_samp=cfg.cate_samp, \
                                                    last_drop=cfg.last_drop, addi_step=cfg.addi_step)
            elif cfg.algorithm == "ode_euler":
                xt, _ = bfnsolver.ode_euler_update(xt, step, cate_samp=cfg.cate_samp, \
                                                    last_drop=cfg.last_drop, addi_step=cfg.addi_step)
            elif cfg.algorithm == "ode_bfnsolver1":
                xt, _ = bfnsolver.ode_bfnsolver1_update(xt, step, last_drop=cfg.last_drop)
            elif cfg.algorithm == "ode_bfnsolver2_single_step":
                xt, _ = bfnsolver.ode_bfnsolver2_single_step_upate(xt, step, last_drop=cfg.last_drop)
            elif cfg.algorithm == "ode_bfnsolver2_multi_step":
                xt, data_pred_last = bfnsolver.ode_bfnsolver2_multi_step_update(xt, step, \
                                                                                        data_pred_last=data_pred_last, \
                                                                                        last_drop=cfg.last_drop)  
            elif cfg.algorithm == "sde_bfnsolver1":
                xt, _ = bfnsolver.sde_bfnsolver1_update(xt, step, last_drop=cfg.last_drop, cate_samp=cfg.cate_samp)
            elif cfg.algorithm == "sde_bfnsolver2_multi_step":
                xt, data_pred_last = bfnsolver.sde_bfnsolver2_multi_step_update(xt, step, \
                                                                                        data_pred_last=data_pred_last, \
                                                                                        last_drop=cfg.last_drop)
            else:
                raise NotImplementedError(cfg.algorithm)

        categorical = TorchCategorical(logits=xt, validate_args=False)
        samples = categorical.mode

        cfg.save_path = f"./samples/text8/{cfg.algorithm}/n_steps={cfg.n_steps}"
        # cfg.save_path = f"./samples/text8/{cfg.algorithm}/eta={cfg.eta}/n_steps={cfg.n_steps}/initial_dist={cfg.initial_dist}/cate_samp={cfg.cate_samp}"
        logging.info(f"save samples to {os.path.dirname(cfg.save_path)}")
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        save_samples(samples, cfg.save_path, idx)
        idx += 1


def save_samples(samples: torch.Tensor, save_path: str, idx: int) -> None:
    logging.info(f"save samples to {save_path}")
    file_mode = "a" if idx > 0 else "w" 
    with open(f"{save_path}.txt", file_mode, encoding="utf-8") as file:
        for line in batch_to_str(samples):
            file.write(f"{line}\n")

if __name__ == "__main__":
    config = OmegaConf.from_cli()
    sample(config)