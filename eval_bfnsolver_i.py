import torch
from tools.fid_score import calculate_fid_given_paths
import os
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image
from tqdm.auto import tqdm
from utils import make_config, make_config, get_nnet
import einops
from bfnsolver_i import ImageBFNSolver, batch_sample
import json
from torch import multiprocessing as mp
import accelerate
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

    
@torch.inference_mode()
def evaluate(cfg: DictConfig):

    train_cfg = make_config(cfg.config_file)
    cfg = OmegaConf.merge(train_cfg.sampling, cfg)

    # based on accelerator
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device} seed: {cfg.seed}')

    
    unet = get_nnet(**train_cfg.net)
    unet = accelerator.prepare(unet)
    state_dict = torch.load(cfg.load_model, map_location='cpu')
    new_state_dict = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    accelerator.unwrap_model(unet).load_state_dict(new_state_dict)
    unet.eval()
    logging.info(f"Loaded model from {cfg.load_model}") 

    logging.info(f"sample with {cfg.algorithm}")
    logging.info(f"number of samples: {cfg.n_samples}, n_steps: {cfg.n_steps}, last_drop: {cfg.last_drop}, eta: {cfg.eta}, initial_dist: {cfg.initial_dist}")

    min_variance = train_cfg.bayesian_flow.min_variance
    bfnsolver = ImageBFNSolver(unet, num_steps=cfg.n_steps, min_variance=min_variance, eta=cfg.eta)
  
    cfg.save_path = f"./eval_samples/cifar/{cfg.algorithm}/n_steps={cfg.n_steps}"
    if accelerator.is_main_process and not os.path.exists(cfg.save_path):
        logging.info(f"save samples to {cfg.save_path}")
        os.makedirs(cfg.save_path)

    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]
    
    idx = 0 
    total_batch_size = cfg.batch_size * accelerator.num_processes
    for _batch_size in tqdm(amortize(cfg.n_samples, total_batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
    
        samples_shape = (cfg.batch_size, train_cfg.data.image_size, train_cfg.data.image_size, train_cfg.data.image_channels)
        
        samples = batch_sample(cfg, train_cfg, unet, bfnsolver, samples_shape, min_variance)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]

        if accelerator.is_main_process:
            for image in samples:
                image  = (image / 2 + 0.5).clamp(0, 1)
                image = einops.rearrange(image, 'H W C -> C H W')
                save_image(image, os.path.join(cfg.save_path, f"{idx}.png"))
                idx += 1
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        cal_fid(cfg.fid_stat, cfg.save_path)


def cal_fid(fid_stat: str, sample_path: str):
    fid = calculate_fid_given_paths((fid_stat, sample_path))
    logging.info(f"fid: {fid}")
    logging.info(f"save result to: {sample_path}-fid.json")

    result = {
        "fid": fid,
        "sample_path": sample_path
    }
    with open(sample_path + '-fid.json', 'w') as f:
        f.write(json.dumps(result))

if __name__ == "__main__":
    evaluate(OmegaConf.from_cli())