import os
import numpy as np

import torch
import torch.random
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from dataloader import OpenCellLoader
from celle import VQGanVAE, CELLE
from omegaconf import OmegaConf
import argparse, os, sys, datetime, glob


from celle.celle import gumbel_sample, top_k

torch.random.manual_seed(42)
np.random.seed(42)

from celle_taming_main import (
    instantiate_from_config,
    nondefault_trainer_args,
    get_parser,
)


class CellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config_file,
        sequence_mode="simple",
        vocab="bert",
        crop_size=256,
        batch_size=1,
        threshold=False,
        text_seq_len=1000,
        num_workers=1,
        **kwargs,
    ):
        super().__init__()

        self.config_file = config_file
        self.protein_sequence_length = 0
        self.image_folders = []
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.sequence_mode = sequence_mode
        self.threshold = threshold
        self.text_seq_len = int(text_seq_len)
        self.vocab = vocab
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

    def setup(self):
        # called on every GPU
        self.cell_dataset_train = OpenCellLoader(
            config_file=self.config_file,
            crop_size=self.crop_size,
            split_key="train",
            crop_method="random",
            sequence_mode=self.sequence_mode,
            vocab=self.vocab,
            text_seq_len=self.text_seq_len,
            threshold=self.threshold,
        )

        self.cell_dataset_val = OpenCellLoader(
            config_file=self.config_file,
            crop_size=self.crop_size,
            crop_method="center",
            split_key="val",
            sequence_mode=self.sequence_mode,
            vocab=self.vocab,
            text_seq_len=self.text_seq_len,
            threshold=self.threshold,
        )

    def prepare_data(self):

        pass

    def train_dataloader(self):
        return DataLoader(
            self.cell_dataset_train,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cell_dataset_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    # def test_dataloader(self):
    #    transforms = ...
    #    return DataLoader(self.test, batch_size=64)


class CELLE_trainer(pl.LightningModule):
    def __init__(
        self,
        vqgan_model_path,
        vqgan_config_path,
        ckpt_path=None,
        image_key="target",
        condition_model_path=None,
        condition_config_path=None,
        num_images=2,
        dim=2,
        num_text_tokens=10000,
        text_seq_len=256,
        depth=1,
        heads=8,
        dim_head=64,
        reversible=False,
        attn_dropout=0.0,
        ff_dropout=0,
        attn_types=None,
        loss_img_weight=7,
        stable=False,
        sandwich_norm=False,
        shift_tokens=True,
        rotary_emb=True,
        text_embedding=None,
        fixed_embedding=False,
        loss_cond_weight=1,
        learning_rate=3e-4,
        monitor="val_loss",
    ):
        super().__init__()

        vae = VQGanVAE(
            vqgan_model_path=vqgan_model_path, vqgan_config_path=vqgan_config_path
        )

        self.image_key = image_key

        if condition_config_path:
            condition_vae = VQGanVAE(
                vqgan_model_path=condition_model_path,
                vqgan_config_path=condition_config_path,
            )
        else:
            condition_vae = None

        self.celle = CELLE(
            dim=dim,
            vae=vae,  # automatically infer (1) image sequence length and (2) number of image tokens
            condition_vae=condition_vae,
            num_images=num_images,
            num_text_tokens=num_text_tokens,  # vocab size for text
            text_seq_len=text_seq_len,  # text sequence length
            depth=depth,  # should aim to be 64
            heads=heads,  # attention heads
            reversible=reversible,  # should aim to be True
            dim_head=dim_head,  # attention head dimension
            attn_dropout=attn_dropout,  # attention dropout
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            loss_img_weight=loss_img_weight,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
            text_embedding=text_embedding,
            fixed_embedding=fixed_embedding,
            loss_cond_weight=loss_cond_weight
            # feedforward dropout
        )

        self.learning_rate = learning_rate
        self.num_text_tokens = num_text_tokens
        self.num_images = num_images

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.celle.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, text, condition, target, return_loss=True):

        return self.celle(
            text=text, condition=condition, image=target, return_loss=return_loss
        )

    def get_input(self, batch):
        text = batch["sequence"].squeeze(1)
        condition = batch["nucleus"]
        target = batch[self.image_key]

        return text, condition, target

    def get_image_from_logits(self, logits, temperature=0.9):

        filtered_logits = top_k(logits, thres=0.5)
        sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

        self.celle.vae.eval()
        out = self.celle.vae.decode(
            sample[:, self.celle.text_seq_len + self.celle.condition_seq_len :]
            - (self.celle.num_text_tokens + self.celle.num_condition_tokens)
        )

        return out

    def get_loss(self, text, condition, target):


        loss_dict = {}

        loss, loss_dict, logits = self(
            text, condition, target, return_loss=True
        )

        return loss, loss_dict

    def total_loss(
        self,
        loss,
        loss_dict={"loss_text": 0, "loss_cond": 0, "loss_img": 0},
        mode="train",
    ):

        loss_dict = {f"{mode}/{key}": value for key, value in loss_dict.items()}

        self.log(
            f"{mode}/loss_text",
            loss_dict[f"{mode}/loss_text"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            f"{mode}/loss_cond",
            loss_dict[f"{mode}/loss_cond"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            f"{mode}/loss_img",
            loss_dict[f"{mode}/loss_img"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch, batch_idx):

        text, condition, target = self.get_input(batch)
        loss, log_dict = self.get_loss(text, condition, target)

        loss = self.total_loss(loss, log_dict, mode="train")

        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            text, condition, target = self.get_input(batch)
            loss, log_dict = self.get_loss(text, condition, target)

            loss = self.total_loss(loss, log_dict, mode="val")

        return loss

    def configure_optimizers(self):


        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def scale_image(self, image):

        for tensor in image:
            if torch.min(tensor) < 0:
                tensor += -torch.min(tensor)
            else:
                tensor -= torch.min(tensor)

            tensor /= torch.max(tensor)

        return image

    @torch.no_grad()
    def log_images(self, batch, **kwargs):

        log = dict()
        text, condition, target = self.get_input(batch)
        text = text.squeeze(1).to(self.device)
        condition = condition.to(self.device)

        out = self.celle.generate_images(text=text, condition=condition, use_cache=True)

        log["condition"] = self.scale_image(condition)
        log["output"] = self.scale_image(out)
        if self.image_key == "threshold":
            log["threshold"] = self.scale_image(target)
            log["target"] = self.scale_image(batch["target"])
        else:
            log["target"] = self.scale_image(target)

        return log


# from https://github.com/CompVis/taming-transformers/blob/master/celle_main.py

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: celle_main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python celle_main.py`
    # (in particular `celle_main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index("logs") + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        # trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                },
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "celle_taming_main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "image_logger": {
                "target": "celle_taming_main.ImageLogger",
                "params": {
                    "batch_frequency": 1500,
                    "max_images": 5,
                    "clamp": False,
                    "increase_log_steps": False,
                },
            },
            # "learning_rate_logger": {
            #     "target": "celle_taming_main.LearningRateMonitor",
            #     "params": {
            #         "logging_interval": "step",
            #         # "log_momentum": True
                # },
            # },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.setup()
        data.prepare_data()

        # configure learning rate
        bs, lr = config.data.params.batch_size, config.model.learning_rate

        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * lr

        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, lr
            )
        )

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)