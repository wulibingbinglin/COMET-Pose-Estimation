import os
import torch
import csv
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from train_util import *
from train_eval_func_new_cp5 import train_or_eval_fn

class CsvLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        with open(self.path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)

def test_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    
    torch.backends.cudnn.benchmark = False 

    accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision=cfg.mixed_precision)
    set_seed_and_print(cfg.seed)

    if accelerator.is_main_process:
        csv_path = os.path.join(cfg.output_dir, "test_results.csv")
        csv_fields = ["epoch", "it", "mode"] + list(TO_PLOT_METRICS) + ["lr"]
        logger = CsvLogger(csv_path, csv_fields)
        accelerator.print(f"Test results will be saved to {csv_path}")

    _, _, _, eval_dataloader = build_dataset(cfg)
    
    if hasattr(eval_dataloader, 'batch_sampler') and eval_dataloader.batch_sampler is not None:
        eval_dataloader.batch_sampler.drop_last = True
    
    accelerator.print(f"Length of eval dataloader: {len(eval_dataloader)}")

    try:
        model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model: {e}")

    optimizer, lr_scheduler = build_optimizer(cfg, model, eval_dataloader)

    start_epoch = 0
    weights_loaded = False
    
    if cfg.train.resume_ckpt and os.path.isfile(cfg.train.resume_ckpt):
        accelerator.print(f"Loading weights from specific file: {cfg.train.resume_ckpt}")
        model = load_model_weights2(model, cfg.train.resume_ckpt, accelerator.device, cfg.relax_load)
        weights_loaded = True

    model, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, eval_dataloader, optimizer, lr_scheduler
    )

    accelerator.print(f"---------- Start Testing (Model Epoch: {start_epoch-1}) ----------")
    
    stats = VizStats(TO_PLOT_METRICS)
    
    lr = lr_scheduler.get_last_lr()[0]
    
    train_or_eval_fn(
        model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, 
        training=False, 
        epoch=start_epoch-1
    )

    stats.update({"lr": lr}, stat_set="eval")
    epoch_data = stats.get_epoch_averages()["eval"]
    
    if accelerator.is_main_process:
        row = {
            "epoch": epoch_data.get("epoch", start_epoch-1),
            "it": epoch_data.get("it", 0),
            "mode": "eval",
        }
        for k in TO_PLOT_METRICS:
            row[k] = epoch_data.get(k, "")
        logger.log(row)
        print(f"✅ Test finished. Results saved to {csv_path}")

    return True

if __name__ == '__main__':
    cfg = OmegaConf.load('test.yaml')
    test_fn(cfg)