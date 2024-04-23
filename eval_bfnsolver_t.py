import os
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical as TorchCategorical
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from tools.text_spelling import calculate_spelling_accuracy
from typing import Tuple, Optional
import json
import logging

from utils import seed_everything, make_config, get_nnet, batch_to_str
from bfnsolver_t import TextBFNSolver

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set global seeds and flags for reproducibility and performance
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
# torch.set_default_dtype(torch.float64)

@torch.inference_mode()
def evaluate(cfg: DictConfig):
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

        cfg.save_path = f"./eval_samples/text8/{cfg.algorithm}/n_steps={cfg.n_steps}"
        # cfg.save_path = f"./samples/text8/{cfg.algorithm}/eta={cfg.eta}/n_steps={cfg.n_steps}/initial_dist={cfg.initial_dist}/cate_samp={cfg.cate_samp}"
        logging.info(f"save samples to {os.path.dirname(cfg.save_path)}")
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        save_samples(samples, cfg.save_path, idx)
        idx += 1

    # eval
    accuracy = calculate_spelling_accuracy(f"{cfg.save_path}.txt")
    logging.info(f"accuracy: {accuracy}")
    logging.info(f"save result to: {cfg.save_path}-acc.json")
    result = {
        "accuracy": accuracy,
        "save_path": cfg.save_path
    }
    with open(cfg.save_path + '-acc.json', 'w') as f:
        f.write(json.dumps(result))

def save_samples(samples: torch.Tensor, save_path: str, idx: int) -> None:
    logging.info(f"save samples to {save_path}")
    file_mode = "a" if idx > 0 else "w" 
    with open(f"{save_path}.txt", file_mode, encoding="utf-8") as file:
        for line in batch_to_str(samples):
            file.write(f"{line}\n")

if __name__ == "__main__":
    config = OmegaConf.from_cli()
    evaluate(config)