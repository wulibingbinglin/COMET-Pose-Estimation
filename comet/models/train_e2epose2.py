import warnings
warnings.filterwarnings("ignore", r"Parameter '.*' is unused\.")
import pickle
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import model_io, vis_utils
from train_util import *
from train_eval_func_new_cp5 import train_or_eval_fn
import psutil
import torch
from datetime import datetime
import os
import csv

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

def log_memory_status(step, location, extra_info=None):
    process = psutil.Process()
    current_memory = process.memory_info().rss
    gpu_memory = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()
    print(f"\n💾 Memory Status at {location}:")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Memory: {current_memory / 1024 ** 3:.2f} GB")
    print(f"GPU Allocated: {gpu_memory / 1024 ** 3:.2f} GB")
    print(f"GPU Reserved: {gpu_reserved / 1024 ** 3:.2f} GB")
    if extra_info:
        print(f"Additional Info: {extra_info}")
    print("-" * 50)
    return current_memory

def train_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision=cfg.mixed_precision)
    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))
    accelerator.print(accelerator.state)

    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = getattr(cfg.train, "cudnnbenchmark", True)

    set_seed_and_print(cfg.seed)

    if accelerator.is_main_process:
        csv_path = os.path.join(cfg.exp_dir, "train_eval_stats.csv")
        csv_fields = ["epoch", "it", "mode"] + list(TO_PLOT_METRICS) + ["lr"]
        logger = CsvLogger(csv_path, csv_fields)

    dataset, eval_dataset, dataloader, eval_dataloader = build_dataset(cfg)
    dataloader.batch_sampler.drop_last = True
    eval_dataloader.batch_sampler.drop_last = True

    try:
        model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)
        model = model.to(accelerator.device)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate the model with error: {e}")

    num_epochs = cfg.train.epochs
    optimizer, lr_scheduler = build_optimizer(cfg, model, dataloader)

    if cfg.train.resume_ckpt:
        accelerator.print(f"Loading ckpt from {cfg.train.resume_ckpt}")
        model = load_model_weights(model, cfg.train.resume_ckpt, accelerator.device, cfg.relax_load)

    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))
    accelerator.print(f"dataloader has {dataloader.num_workers} num_workers")

    start_epoch = 0
    stats = VizStats(TO_PLOT_METRICS)

    if cfg.train.auto_resume:
        last_checkpoint = find_last_checkpoint(cfg.exp_dir)
        try:
            resume_epoch = int(os.path.basename(last_checkpoint)[5:])
        except:
            resume_epoch = -1
        print("last_checkpoint=",last_checkpoint)
        if last_checkpoint is not None and resume_epoch > 0:
            accelerator.print(f"Loading ckpt from {last_checkpoint}")
            accelerator.load_state(last_checkpoint)
            try:
                loaded_tdict = pickle.load(open(os.path.join(last_checkpoint, "tdict.pkl"), "rb"))
                start_epoch = loaded_tdict["epoch"]+1
            except:
                start_epoch = resume_epoch+1
            try:
                stats = stats.load(os.path.join(last_checkpoint, "train_stats.jgz"))
            except:
                stats.hard_reset(epoch=start_epoch)
                accelerator.print(f"No stats to load from {last_checkpoint}")
        else:
            accelerator.print(f"Starting from scratch")

    accelerator.print(f"----------Start to eval at epoch {start_epoch-1}----------")
    lr = lr_scheduler.get_last_lr()[0]
    print("lr",lr)
    train_or_eval_fn(
        model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=False, epoch=start_epoch-1
    )
    stats.update({"lr": lr}, stat_set="eval")
    epoch_data = stats.get_epoch_averages()["eval"]
    row = {
        "epoch": epoch_data["epoch"],
        "it": epoch_data["it"],
        "mode": "eval",
    }
    for k in TO_PLOT_METRICS:
        row[k] = epoch_data.get(k, "")
    logger.log(row)
    return

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()
        set_seed_and_print(cfg.seed + epoch * 1000)
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats,
            accelerator, lr_scheduler, training=True, epoch=epoch
        )
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            stats.update({"lr": lr}, stat_set="train")
            epoch_data = stats.get_epoch_averages()["train"]
            row = {
                "epoch": epoch_data["epoch"],
                "it": epoch_data["it"],
                "mode": "train",
                "lr": epoch_data["lr"],
            }
            for k in TO_PLOT_METRICS:
                row[k] = epoch_data.get(k, "")
            logger.log(row)

        if (epoch != 0) and epoch % cfg.train.ckpt_interval == 0:
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------")
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)
                pickle.dump({"epoch": epoch, "cfg": cfg}, open(os.path.join(ckpt_path, "tdict.pkl"), "wb"))
                stats.save(os.path.join(ckpt_path, "train_stats.jgz"))

        if (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            train_or_eval_fn(
                model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=False,
                epoch=epoch
            )
            stats.update({"lr": lr}, stat_set="eval")
            epoch_data = stats.get_epoch_averages()["eval"]
            row = {
                "epoch": epoch_data["epoch"],
                "it": epoch_data["it"],
                "mode": "eval",
            }
            for k in TO_PLOT_METRICS:
                row[k] = epoch_data.get(k, "")
            logger.log(row)
        else:
            accelerator.print(f"----------Skip the test/eval at epoch {epoch}----------")

    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"), safe_serialization=False)
    return True

if __name__ == '__main__':
    cfg = OmegaConf.load('train.yaml')
    train_fn(cfg)