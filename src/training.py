"""
Main training module for dynamics and CLF models using Hydra and W&B.
"""

import os
import sys
import random
import hydra
import torch
import torch.nn as nn
import numpy as np
import wandb
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamics import ControlAffineNetwork, DynamicsEnsemble
from models.clf import CLFNetwork
from solvers.clf_qp_solver import CLFQPSolver
from data.data_module import DynamicsDataModule, CLFDataModule

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


def get_space_dimension(space) -> int:
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _get_device(accelerator: str = "auto", devices: Any = "auto") -> torch.device:
    """Determine the best available device from accelerator config."""
    if accelerator in ("gpu", "cuda") or (accelerator == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _init_wandb_run(
    config: DictConfig,
    name: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
    log_model: Optional[bool] = None,
    notes: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialise a W&B run from Hydra config.
    All OmegaConf containers are converted to plain Python types before passing
    to wandb to avoid serialisation failures.
    """
    tags = list(config.logger.tags) if config.logger.tags else []
    if extra_tags:
        tags = tags + list(extra_tags)

    run = wandb.init(
        project=str(config.logger.project),
        name=str(name or config.logger.name),
        dir=str(config.logger.save_dir),
        notes=str(notes or config.logger.notes),
        tags=tags,
        group=str(config.logger.group),
        mode="offline" if bool(config.logger.offline) else "online",
        reinit=True,
    )
    return run


# ── Custom Trainer ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Lightweight training loop replacing PyTorch Lightning's Trainer.
    Supports gradient clipping, epoch-end validation, early stopping,
    checkpointing, and W&B logging.
    """

    def __init__(
        self,
        max_epochs: int,
        min_epochs: int = 1,
        gradient_clip_val: float = 0.0,
        val_check_interval: float = 1.0,
        wandb_run=None,
        early_stopping: Optional[Dict[str, Any]] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        device: torch.device = None,
        log_every_n_steps: int = 1,
    ) -> None:
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.gradient_clip_val = gradient_clip_val
        self.val_check_interval = val_check_interval
        self._wandb_run = wandb_run
        self.device = device or torch.device("cpu")
        self.log_every_n_steps = log_every_n_steps

        # Early stopping config
        self._es = None
        if early_stopping and early_stopping.get("enabled", False):
            self._es = {
                "patience": int(early_stopping.get("patience", 10)),
                "min_delta": float(early_stopping.get("min_delta", 1e-4)),
                "mode": str(early_stopping.get("mode", "min")),
                "counter": 0,
                "best": float("inf") if early_stopping.get("mode", "min") == "min" else float("-inf"),
            }

        # Checkpoint config
        self._ckpt = checkpoint or {}
        self._best_val_loss: float = float("inf")
        self._best_model_path: Optional[str] = None
        self._ckpt_save_counter: int = 0

    def _to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    def _run_validation(self, model: nn.Module, data_module) -> Optional[float]:
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_module.val_dataloader()):
                batch = self._to_device(batch, self.device)
                outputs = model.validation_step(batch, batch_idx)
                if isinstance(outputs, dict):
                    v = outputs.get("val_loss", outputs.get("loss", None))
                else:
                    v = outputs
                if v is not None:
                    val_losses.append(v.item() if torch.is_tensor(v) else float(v))
        model.train()
        return sum(val_losses) / len(val_losses) if val_losses else None

    def _run_test(self, model: nn.Module, data_module) -> List[Dict]:
        model.eval()
        results: List[Dict] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_module.test_dataloader()):
                batch = self._to_device(batch, self.device)
                outputs = model.test_step(batch, batch_idx)
                if isinstance(outputs, dict):
                    results.append({
                        k: v.item() if torch.is_tensor(v) else v
                        for k, v in outputs.items()
                    })
        model.train()
        return results

    def _check_early_stop(self, val_loss: Optional[float], epoch: int) -> bool:
        if self._es is None or val_loss is None:
            return False
        if epoch < self.min_epochs:
            return False
        es = self._es
        if es["mode"] == "min":
            improved = val_loss < es["best"] - es["min_delta"]
        else:
            improved = val_loss > es["best"] + es["min_delta"]
        if improved:
            es["best"] = val_loss
            es["counter"] = 0
        else:
            es["counter"] += 1
        return es["counter"] >= es["patience"]

    def _maybe_save_checkpoint(self, model: nn.Module, val_loss: Optional[float], epoch: int) -> None:
        if not self._ckpt:
            return
        dirpath = str(self._ckpt.get("dirpath", "checkpoints"))
        os.makedirs(dirpath, exist_ok=True)

        every_n = int(self._ckpt.get("every_n_epochs", 1))
        if epoch % every_n != 0 and epoch != self.max_epochs - 1:
            return

        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            filename = str(self._ckpt.get("filename", "best_model"))
            path = os.path.join(dirpath, f"{filename}.pt")
            if hasattr(model, "save_checkpoint"):
                model.save_checkpoint(path, val_loss=val_loss)
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                }, path)
            self._best_model_path = path
            print(f"  Saved checkpoint (val_loss={val_loss:.6f}): {path}")

    @property
    def best_model_path(self) -> Optional[str]:
        return self._best_model_path

    def fit(self, model: nn.Module, data_module) -> None:
        """Run the training loop."""
        # Determine starting epoch: resume from model's last trained epoch
        start_epoch = getattr(model, "_next_epoch", 0)

        # Wire up W&B run and device on the model
        model._wandb_run = self._wandb_run
        model.to(self.device)

        optimizer = model.configure_optimizers()

        # on_train_start
        if hasattr(model, "on_train_start"):
            model.on_train_start()

        for epoch in range(start_epoch, self.max_epochs):
            model.current_epoch = epoch
            model.train()

            # on_train_epoch_start
            if hasattr(model, "on_train_epoch_start"):
                model.on_train_epoch_start()

            train_losses: List[float] = []

            for batch_idx, batch in enumerate(data_module.train_dataloader()):
                batch = self._to_device(batch, self.device)
                optimizer.zero_grad()

                outputs = model.training_step(batch, batch_idx)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                loss.backward()

                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

                optimizer.step()
                model.global_step += 1

                if torch.is_tensor(loss):
                    train_losses.append(loss.item())

                if hasattr(model, "on_train_batch_end"):
                    model.on_train_batch_end(outputs, batch, batch_idx)

                if self._wandb_run and model.global_step % self.log_every_n_steps == 0:
                    self._wandb_run.log(
                        {"train_loss_step": loss.item() if torch.is_tensor(loss) else float(loss)},
                        step=model.global_step,
                    )

            # Epoch-level train logging
            if train_losses and self._wandb_run:
                self._wandb_run.log(
                    {"train_loss": sum(train_losses) / len(train_losses), "epoch": epoch},
                    step=model.global_step,
                )

            # on_train_epoch_end
            if hasattr(model, "on_train_epoch_end"):
                model.on_train_epoch_end()

            # End-of-epoch validation
            val_loss = self._run_validation(model, data_module)
            if val_loss is not None and self._wandb_run:
                self._wandb_run.log({"val_loss": val_loss, "epoch": epoch}, step=model.global_step)
            print(f"  Epoch {epoch:4d}/{self.max_epochs-1}  train={sum(train_losses)/len(train_losses) if train_losses else float('nan'):.6f}  val={val_loss if val_loss is not None else float('nan'):.6f}")

            self._maybe_save_checkpoint(model, val_loss, epoch)

            if self._check_early_stop(val_loss, epoch):
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        # Persist epoch pointer for next fit() call (incremental training)
        model._next_epoch = self.max_epochs
        model.current_epoch = self.max_epochs - 1

    def test(self, model: nn.Module, data_module) -> List[Dict]:
        """Run evaluation on the test set."""
        model._wandb_run = self._wandb_run
        model.to(self.device)
        results = self._run_test(model, data_module)
        if results and self._wandb_run:
            agg = {}
            for r in results:
                for k, v in r.items():
                    agg.setdefault(k, []).append(v)
            means = {k: sum(vs) / len(vs) for k, vs in agg.items()}
            self._wandb_run.log({f"test_{k}": v for k, v in means.items()})
        return results


# ── Callback config helpers ────────────────────────────────────────────────────

def _make_trainer(
    config: DictConfig,
    max_epochs: int,
    min_epochs: int = 1,
    wandb_run=None,
    log_every_n_steps: int = 1,
    inference_mode: bool = True,  # kept for signature compat; no-op in custom Trainer
    device: torch.device = None,
) -> Trainer:
    """Build a Trainer from Hydra config."""
    es_cfg = OmegaConf.to_container(config.training.early_stopping, resolve=True)
    ckpt_cfg = OmegaConf.to_container(config.checkpoint, resolve=True)
    return Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        gradient_clip_val=float(config.training.gradient_clip_val),
        val_check_interval=float(config.training.val_check_interval),
        wandb_run=wandb_run,
        early_stopping=es_cfg,
        checkpoint=ckpt_cfg,
        device=device or _get_device(config.device.accelerator, config.device.devices),
        log_every_n_steps=log_every_n_steps,
    )


# ── Training functions ─────────────────────────────────────────────────────────

def train_dynamics_model(config: DictConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a dynamics model incrementally over collected trajectories."""
    set_seed(config.seed)
    device = _get_device(config.device.accelerator, config.device.devices)

    wandb_run = _init_wandb_run(config)

    env = None
    if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
        from data_collection import collect_trajectory, process_trajectory
        print(f"Setting up environment: {config.experiment.env_name}")
        env = gym.make(config.experiment.env_name)

    num_trajectories = config.experiment.data.get('num_trajectories', 20)
    traj_length = config.experiment.data.get('traj_length', 200)
    buffer_size = config.experiment.data.get('buffer_size', num_trajectories)
    epochs_per_trajectory = config.experiment.data.get('epochs_per_trajectory', 5)
    batch_size = config.experiment.training.get("batch_size", config.training.batch_size)
    log_every_n_steps = config.experiment.training.get(
        "log_every_n_steps", max(1, epochs_per_trajectory // 2)
    )

    state_dim = None
    action_dim = None
    model = None
    trajectory_buffer = []
    total_epochs_trained = 0
    trainer = None

    for traj_idx in range(num_trajectories):
        print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} ===")

        if env is not None:
            new_trajectory = collect_trajectory(env, length=traj_length, random_seed=config.seed + traj_idx)
            trajectory_buffer.append(new_trajectory)
            if len(trajectory_buffer) > buffer_size:
                trajectory_buffer.pop(0)
            states, actions, next_states = process_trajectory(trajectory_buffer)
            print(f"Dataset size after trajectory {traj_idx+1}: {states.shape[0]} samples")
        else:
            raise ValueError("Neither environment nor data path specified for training")

        data_module = DynamicsDataModule(
            batch_size=batch_size,
            num_workers=config.experiment.training.get("num_workers", 4),
            normalize=config.experiment.dynamics.get("normalize_data", True),
            random_seed=config.seed,
            states=states,
            actions=actions,
            next_states=next_states,
            persistent_workers=config.experiment.data.get("persistent_workers", False),
            train_ratio=config.experiment.data.get("train_ratio", 0.8),
            val_ratio=config.experiment.data.get("val_ratio", 0.1),
            test_ratio=config.experiment.data.get("test_ratio", 0.1),
        )
        data_module.prepare_data()
        data_module.setup()

        if model is None:
            if env is not None:
                state_dim = get_space_dimension(env.observation_space)
                action_dim = get_space_dimension(env.action_space)
            else:
                batch = next(iter(data_module.train_dataloader()))
                state_dim = batch["states"].shape[1]
                action_dim = batch["actions"].shape[1]

            config_dict = OmegaConf.to_container(config.model, resolve=True)
            config_dict.update({"state_dim": state_dim, "action_dim": action_dim})
            model_config = OmegaConf.create(config_dict)
            model = hydra.utils.instantiate(model_config)

            wandb_run.config.update({
                "state_dim": state_dim,
                "action_dim": action_dim,
                "epochs_per_trajectory": epochs_per_trajectory,
                "buffer_size": buffer_size,
            })
            print(f"Inferred dimensions: state_dim={state_dim}, action_dim={action_dim}")

        trainer = _make_trainer(
            config,
            max_epochs=total_epochs_trained + epochs_per_trajectory,
            min_epochs=total_epochs_trained + 1,
            wandb_run=wandb_run,
            log_every_n_steps=log_every_n_steps,
            device=device,
        )
        print(f"Training for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
        trainer.fit(model, data_module)
        total_epochs_trained += epochs_per_trajectory

        wandb_run.log({
            "trajectory_completed": traj_idx + 1,
            "total_epochs_trained": total_epochs_trained,
        })

    print("\n=== Testing final model ===")
    test_results = trainer.test(model, data_module) if trainer is not None else []

    best_model_path = trainer.best_model_path if trainer is not None else None
    print(f"Best model path: {best_model_path}")
    wandb_run.summary.update({"best_model_path": str(best_model_path or "")})

    if env is not None:
        env.close()

    wandb_run.finish()
    return model, {"test_results": test_results, "best_model_path": best_model_path}


def train_clf_model(
    config: DictConfig,
    dynamics_model: Optional[nn.Module] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a CLF model."""
    set_seed(config.seed)
    device = _get_device(config.device.accelerator, config.device.devices)

    wandb_run = _init_wandb_run(config)

    env = None
    states = None
    state_dim = None
    action_dim = None
    action_low = None
    action_high = None

    if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
        from data_collection import collect_trajectory, process_trajectory

        print(f"Setting up environment for CLF training: {config.experiment.env_name}")
        env = gym.make(config.experiment.env_name)
        state_dim = get_space_dimension(env.observation_space)
        action_dim = get_space_dimension(env.action_space)
        action_low = float(env.action_space.low.min())
        action_high = float(env.action_space.high.max())
        print(f"Inferred from environment: state_dim={state_dim}, action_dim={action_dim}, "
              f"action_limits=({action_low}, {action_high})")

        num_trajectories = config.experiment.data.get('num_trajectories', 5)
        traj_length = config.experiment.data.get('traj_length', 50)
        trajectory_buffer = []

        for traj_idx in range(num_trajectories):
            print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} for CLF training ===")
            new_trajectory = collect_trajectory(env, length=traj_length, random_seed=config.seed + traj_idx)
            trajectory_buffer.append(new_trajectory)

        states, _, _ = process_trajectory(trajectory_buffer)
        print(f"Generated dataset for CLF training: {states.shape[0]} state samples")
        env.close()

    if states is None:
        raise ValueError(
            "No state data available for CLF training. Set config.experiment.env_name."
        )

    data_module = CLFDataModule(
        states=states,
        batch_size=config.experiment.training.get("batch_size", config.training.batch_size),
        num_workers=config.experiment.training.get("num_workers", 4),
        normalize=config.experiment.dynamics.get("normalize_data", True),
        random_seed=config.seed,
        persistent_workers=config.experiment.data.get("persistent_workers", False),
        train_ratio=config.experiment.data.get("train_ratio", 0.8),
        val_ratio=config.experiment.data.get("val_ratio", 0.1),
        test_ratio=config.experiment.data.get("test_ratio", 0.1),
    )
    if hasattr(config.experiment.data, "data_path") and config.experiment.data.data_path:
        data_module.data_path = config.experiment.data.data_path

    data_module.prepare_data()
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    if state_dim is None:
        state_dim = batch["states"].shape[1]

    clf_exp = getattr(config.experiment, "clf", OmegaConf.create({}))
    exp_const = float(getattr(clf_exp, "exp_const", 1.0))
    config_dict = {
        "_target_": "src.models.clf.CLFNetwork",
        "state_dim": state_dim,
        "hidden_dim": int(getattr(clf_exp, "hidden_dim", 64)),
        "exp_const": exp_const,
    }
    model_config = OmegaConf.create(config_dict)
    model = hydra.utils.instantiate(model_config)

    model.dynamics_model = dynamics_model

    if action_dim is not None:
        model.qp_solver = CLFQPSolver(
            action_dim=action_dim,
            action_limits=(action_low, action_high),
            exp_const=exp_const,
        )

    if hasattr(model, 'equilibrium'):
        try:
            from pendulum_utils import get_pendulum_equilibrium
            model.equilibrium.copy_(get_pendulum_equilibrium())
        except Exception:
            pass

    print(f"Inferred dimensions: state_dim={state_dim}" +
          (f", action_dim={action_dim}, action_limits=({action_low}, {action_high})"
           if action_dim is not None else ""))

    trainer = _make_trainer(
        config,
        max_epochs=config.training.max_epochs,
        wandb_run=wandb_run,
        inference_mode=False,  # CLF gradient() needs grad tracking
        device=device,
    )
    trainer.fit(model, data_module)
    test_results = trainer.test(model, data_module)

    best_model_path = trainer.best_model_path
    print(f"Best model path: {best_model_path}")
    wandb_run.summary.update({"best_model_path": str(best_model_path or "")})

    wandb_run.finish()
    return model, {"test_results": test_results, "best_model_path": best_model_path}


def evaluate_control_policy(
    config: DictConfig,
    clf_model: nn.Module,
    dynamics_model: nn.Module
) -> Dict[str, Any]:
    """Evaluate the CLF-QP control policy."""
    if hasattr(clf_model, 'qp_solver') and clf_model.qp_solver is not None:
        qp_solver = CLFQPSolver(
            action_dim=clf_model.qp_solver.action_dim,
            action_limits=(clf_model.qp_solver.action_lower, clf_model.qp_solver.action_upper),
            exp_const=clf_model.qp_solver.const,
        )
    else:
        qp_solver = CLFQPSolver(action_dim=dynamics_model.action_dim)

    wandb_run = _init_wandb_run(
        config,
        name=f"{config.logger.name}_eval",
        extra_tags=["evaluation"],
        log_model=False,
        notes=f"Evaluation of {config.logger.name}",
    )

    # Supply W&B run for standalone evaluation visualizations (S5 pattern)
    qp_solver._eval_logger = wandb_run

    if hasattr(config.experiment, "eval_grid") and config.experiment.eval_grid:
        x_range = torch.linspace(-3, 3, 20)
        y_range = torch.linspace(-3, 3, 20)
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        states = torch.stack([X.flatten(), Y.flatten()], dim=1)
        if dynamics_model.state_dim > 2:
            zeros = torch.zeros(len(states), dynamics_model.state_dim - 2)
            states = torch.cat([states, zeros], dim=1)
    else:
        n_samples = config.experiment.get("n_eval_samples", 1000)
        states = torch.randn(n_samples, dynamics_model.state_dim) * 2.0

    device = next(clf_model.parameters()).device
    states = states.to(device)

    results = qp_solver.solve_batch(
        states, clf_model, dynamics_model,
        batch_size=config.training.batch_size,
        log_results=True
    )

    if hasattr(config.experiment, "visualize_admissible_controls") and config.experiment.visualize_admissible_controls:
        state_norms = torch.norm(states, dim=1)
        close_idx = torch.where((state_norms > 0.5) & (state_norms < 1.0))[0]
        mid_idx = torch.where((state_norms > 1.5) & (state_norms < 2.0))[0]
        far_idx = torch.where(state_norms > 2.5)[0]

        sample_states = []
        if len(close_idx) > 0:
            sample_states.append(states[close_idx[0]])
        if len(mid_idx) > 0:
            sample_states.append(states[mid_idx[0]])
        if len(far_idx) > 0:
            sample_states.append(states[far_idx[0]])

        for state in sample_states:
            qp_solver.compute_admissible_control_set(state, clf_model, dynamics_model, num_samples=1000, log_results=True)

    wandb_run.finish()
    return results


# ── Hydra main ─────────────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig) -> Dict[str, Any]:
    print(OmegaConf.to_yaml(config))

    results = {}

    if config.experiment.task == "dynamics_learning":
        model, train_results = train_dynamics_model(config)
        results["dynamics"] = train_results

    elif config.experiment.task == "clf_training":
        dynamics_model = None
        if "dynamics_model_path" in config.experiment and config.experiment.dynamics_model_path:
            print(f"Loading dynamics model from {config.experiment.dynamics_model_path}")
            dynamics_model = DynamicsEnsemble.load_checkpoint(config.experiment.dynamics_model_path)
            dynamics_model.eval()

        clf_model, train_results = train_clf_model(config, dynamics_model)
        results["clf"] = train_results

        if dynamics_model is not None and config.experiment.get("evaluate_control", False):
            eval_results = evaluate_control_policy(config, clf_model, dynamics_model)
            results["control_evaluation"] = eval_results

    elif config.experiment.task == "full_pipeline":
        env = None
        if hasattr(config.experiment, 'env_name') and config.experiment.env_name is not None:
            from data_collection import collect_trajectory, process_trajectory
            print(f"Setting up environment: {config.experiment.env_name}")
            env = gym.make(config.experiment.env_name)
            action_low = float(env.action_space.low.min())
            action_high = float(env.action_space.high.max())
        else:
            raise ValueError("Environment name must be provided for full pipeline")

        num_trajectories = config.experiment.data.get('num_trajectories', 5)
        traj_length = config.experiment.data.get('traj_length', 50)
        buffer_size = config.experiment.data.get('buffer_size', 3)
        epochs_per_trajectory = config.experiment.data.get('epochs_per_trajectory', 5)
        batch_size = config.experiment.training.get("batch_size", config.training.batch_size)
        log_every_n_steps = config.experiment.training.get("log_every_n_steps", 2)

        device = _get_device(config.device.accelerator, config.device.devices)
        trajectory_buffer = []
        dynamics_model = None
        clf_model = None

        dynamics_run = _init_wandb_run(
            config,
            name=f"{config.logger.name}_dynamics",
            extra_tags=["dynamics"],
            notes=f"{config.logger.notes} - Dynamics",
        )
        clf_run = _init_wandb_run(
            config,
            name=f"{config.logger.name}_clf",
            extra_tags=["clf"],
            notes=f"{config.logger.notes} - CLF",
        )

        dynamics_epochs_trained = 0
        clf_epochs_trained = 0
        dynamics_trainer = None
        clf_trainer = None

        for traj_idx in range(num_trajectories):
            print(f"\n=== Collecting trajectory {traj_idx+1}/{num_trajectories} ===")
            new_trajectory = collect_trajectory(env, length=traj_length, random_seed=config.seed + traj_idx)
            trajectory_buffer.append(new_trajectory)
            if len(trajectory_buffer) > buffer_size:
                trajectory_buffer.pop(0)
            states, actions, next_states = process_trajectory(trajectory_buffer)
            print(f"Dataset size after trajectory {traj_idx+1}: {states.shape[0]} samples")

            # ── Dynamics training ────────────────────────────────────────────
            dynamics_data_module = DynamicsDataModule(
                states=states, actions=actions, next_states=next_states,
                batch_size=batch_size,
                num_workers=config.experiment.training.get("num_workers", 4),
                normalize=config.experiment.dynamics.get("normalize_data", True),
                random_seed=config.seed,
                persistent_workers=config.experiment.data.get("persistent_workers", False),
                train_ratio=config.experiment.data.get("train_ratio", 0.8),
                val_ratio=config.experiment.data.get("val_ratio", 0.1),
                test_ratio=config.experiment.data.get("test_ratio", 0.1),
            )
            dynamics_data_module.prepare_data()
            dynamics_data_module.setup()

            if dynamics_model is None:
                if env is not None:
                    state_dim = get_space_dimension(env.observation_space)
                    action_dim = get_space_dimension(env.action_space)
                else:
                    batch = next(iter(dynamics_data_module.train_dataloader()))
                    state_dim = batch["states"].shape[1]
                    action_dim = batch["actions"].shape[1]

                dynamics_config = OmegaConf.to_container(config.model, resolve=True)
                dynamics_config.update({"state_dim": state_dim, "action_dim": action_dim})
                dynamics_model = hydra.utils.instantiate(OmegaConf.create(dynamics_config))

                dynamics_run.config.update({
                    "state_dim": state_dim, "action_dim": action_dim,
                    "epochs_per_trajectory": epochs_per_trajectory,
                    "buffer_size": buffer_size,
                })
                print(f"Inferred dimensions: state_dim={state_dim}, action_dim={action_dim}")

            dynamics_trainer = _make_trainer(
                config,
                max_epochs=dynamics_epochs_trained + epochs_per_trajectory,
                min_epochs=dynamics_epochs_trained + 1,
                wandb_run=dynamics_run,
                log_every_n_steps=log_every_n_steps,
                device=device,
            )
            print(f"Training dynamics model for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
            dynamics_trainer.fit(dynamics_model, dynamics_data_module)
            dynamics_epochs_trained += epochs_per_trajectory

            # ── CLF training ─────────────────────────────────────────────────
            clf_data_module = CLFDataModule(
                states=states,
                batch_size=batch_size,
                num_workers=config.experiment.training.get("num_workers", 4),
                normalize=config.experiment.dynamics.get("normalize_data", True),
                random_seed=config.seed,
                persistent_workers=config.experiment.data.get("persistent_workers", False),
                train_ratio=config.experiment.data.get("train_ratio", 0.8),
                val_ratio=config.experiment.data.get("val_ratio", 0.1),
                test_ratio=config.experiment.data.get("test_ratio", 0.1),
            )
            clf_data_module.prepare_data()
            clf_data_module.setup()

            if clf_model is None:
                clf_config_dict = {
                    "_target_": "src.models.clf.CLFNetwork",
                    "state_dim": state_dim,
                    "hidden_dim": config.experiment.clf.get("hidden_dim", 64),
                    "dropout_rate": 0.1,
                    "learning_rate": 0.001,
                    "exp_const": float(config.experiment.clf.get("exp_const", 1.0)),
                }
                clf_model = hydra.utils.instantiate(OmegaConf.create(clf_config_dict))
                clf_model.dynamics_model = dynamics_model
                clf_model.qp_solver = CLFQPSolver(
                    action_dim=action_dim,
                    action_limits=(action_low, action_high),
                    exp_const=float(config.experiment.clf.get("exp_const", 1.0)),
                )
                if hasattr(clf_model, 'equilibrium'):
                    try:
                        from pendulum_utils import get_pendulum_equilibrium
                        clf_model.equilibrium.copy_(get_pendulum_equilibrium())
                    except Exception:
                        pass
                clf_run.config.update(clf_config_dict)

            clf_trainer = _make_trainer(
                config,
                max_epochs=clf_epochs_trained + epochs_per_trajectory,
                min_epochs=clf_epochs_trained + 1,
                wandb_run=clf_run,
                log_every_n_steps=log_every_n_steps,
                inference_mode=False,
                device=device,
            )
            print(f"Training CLF model for {epochs_per_trajectory} epochs on trajectory {traj_idx+1}")
            clf_trainer.fit(clf_model, clf_data_module)
            clf_epochs_trained += epochs_per_trajectory

            dynamics_run.log({"trajectory_completed": traj_idx + 1, "total_epochs_trained": dynamics_epochs_trained})
            clf_run.log({"trajectory_completed": traj_idx + 1, "total_epochs_trained": clf_epochs_trained})

        # Test final models
        print("\n=== Testing final dynamics model ===")
        dynamics_test_results = dynamics_trainer.test(dynamics_model, dynamics_data_module) if dynamics_trainer else []

        print("\n=== Testing final CLF model ===")
        clf_test_results = clf_trainer.test(clf_model, clf_data_module) if clf_trainer else []

        if config.experiment.get("evaluate_control", False):
            print("\n=== Evaluating control policy ===")
            eval_results = evaluate_control_policy(config, clf_model, dynamics_model)
            results["control_evaluation"] = eval_results

        results["dynamics"] = {"test_results": dynamics_test_results}
        results["clf"] = {"test_results": clf_test_results}

        if dynamics_trainer and dynamics_trainer.best_model_path:
            print(f"Best dynamics model path: {dynamics_trainer.best_model_path}")
            dynamics_run.summary.update({"best_model_path": dynamics_trainer.best_model_path})
            results["dynamics"]["best_model_path"] = dynamics_trainer.best_model_path

        if clf_trainer and clf_trainer.best_model_path:
            print(f"Best CLF model path: {clf_trainer.best_model_path}")
            clf_run.summary.update({"best_model_path": clf_trainer.best_model_path})
            results["clf"]["best_model_path"] = clf_trainer.best_model_path

        dynamics_run.finish()
        clf_run.finish()

        if env is not None:
            env.close()

    else:
        raise ValueError(f"Unknown task: {config.experiment.task}")

    return results


if __name__ == "__main__":
    main()
