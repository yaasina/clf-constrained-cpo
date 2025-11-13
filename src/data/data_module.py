"""
Data module for loading and processing datasets for dynamics learning and CLF training.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


class DynamicsDataset(Dataset):
    """
    Dataset for dynamics learning containing state transition triplets: (state, action, next_state).
    Can optionally include state derivatives for direct dynamics learning.
    """
    
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        state_derivatives: Optional[torch.Tensor] = None,
        normalize: bool = False
    ) -> None:
        """
        Initialize the dataset with state transition data.
        
        Args:
            states: Tensor of states [n_samples, state_dim]
            actions: Tensor of actions [n_samples, action_dim]
            next_states: Tensor of next states [n_samples, state_dim]
            state_derivatives: Optional tensor of state derivatives [n_samples, state_dim]
            normalize: Whether to normalize the data
        """
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.state_derivatives = state_derivatives
        
        # Compute normalization statistics if needed
        if normalize:
            self.normalize_data()
        else:
            self.state_mean = None
            self.state_std = None
            self.action_mean = None
            self.action_std = None
            self.is_normalized = False
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing states, actions, next_states, and optionally state_derivatives
        """
        sample = {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "next_states": self.next_states[idx]
        }
        
        if self.state_derivatives is not None:
            sample["state_derivatives"] = self.state_derivatives[idx]
            
        return sample
    
    def normalize_data(self) -> None:
        """Normalize the data to have zero mean and unit variance."""
        # Compute statistics
        self.state_mean = torch.mean(self.states, dim=0, keepdim=True)
        self.state_std = torch.std(self.states, dim=0, keepdim=True)
        self.action_mean = torch.mean(self.actions, dim=0, keepdim=True)
        self.action_std = torch.std(self.actions, dim=0, keepdim=True)
        
        # Replace any zero std with 1 to avoid division by zero
        self.state_std[self.state_std < 1e-8] = 1.0
        self.action_std[self.action_std < 1e-8] = 1.0
        
        # Normalize the data
        self.states = (self.states - self.state_mean) / self.state_std
        self.actions = (self.actions - self.action_mean) / self.action_std
        self.next_states = (self.next_states - self.state_mean) / self.state_std
        
        if self.state_derivatives is not None:
            # Normalize derivatives by the state std (to maintain relationship to states)
            self.state_derivatives = self.state_derivatives / self.state_std
        
        self.is_normalized = True
    
    def denormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Denormalize states back to the original scale.
        
        Args:
            states: Normalized states [n_samples, state_dim]
            
        Returns:
            Denormalized states [n_samples, state_dim]
        """
        if not self.is_normalized:
            return states
        
        return states * self.state_std + self.state_mean
    
    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions back to the original scale.
        
        Args:
            actions: Normalized actions [n_samples, action_dim]
            
        Returns:
            Denormalized actions [n_samples, action_dim]
        """
        if not self.is_normalized:
            return actions
        
        return actions * self.action_std + self.action_mean
    
    def denormalize_state_derivatives(self, state_derivatives: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state derivatives back to the original scale.
        
        Args:
            state_derivatives: Normalized state derivatives [n_samples, state_dim]
            
        Returns:
            Denormalized state derivatives [n_samples, state_dim]
        """
        if not self.is_normalized:
            return state_derivatives
        
        return state_derivatives * self.state_std
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get the normalization statistics.
        
        Returns:
            Dictionary containing state_mean, state_std, action_mean, action_std
        """
        if not self.is_normalized:
            return {
                "state_mean": None,
                "state_std": None,
                "action_mean": None,
                "action_std": None
            }
        
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std
        }


class CLFDataset(Dataset):
    """
    Dataset for CLF learning containing states and optionally a dynamics model for Lie derivative computation.
    """
    
    def __init__(
        self,
        states: torch.Tensor,
        dynamics_model: Optional[torch.nn.Module] = None,
        normalize: bool = False
    ) -> None:
        """
        Initialize the dataset with states and optionally a dynamics model.
        
        Args:
            states: Tensor of states [n_samples, state_dim]
            dynamics_model: Optional dynamics model for Lie derivative computation
            normalize: Whether to normalize the data
        """
        self.states = states
        self.dynamics_model = dynamics_model
        
        # Compute normalization statistics if needed
        if normalize:
            self.normalize_data()
        else:
            self.state_mean = None
            self.state_std = None
            self.is_normalized = False
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing states (the dynamics model is stored at the dataset level)
        """
        sample = {
            "states": self.states[idx]
        }
            
        return sample
    
    def normalize_data(self) -> None:
        """Normalize the data to have zero mean and unit variance."""
        # Compute statistics
        self.state_mean = torch.mean(self.states, dim=0, keepdim=True)
        self.state_std = torch.std(self.states, dim=0, keepdim=True)
        
        # Replace any zero std with 1 to avoid division by zero
        self.state_std[self.state_std < 1e-8] = 1.0
        
        # Normalize the data
        self.states = (self.states - self.state_mean) / self.state_std
        
        self.is_normalized = True
    
    def denormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Denormalize states back to the original scale.
        
        Args:
            states: Normalized states [n_samples, state_dim]
            
        Returns:
            Denormalized states [n_samples, state_dim]
        """
        if not self.is_normalized:
            return states
        
        return states * self.state_std + self.state_mean
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get the normalization statistics.
        
        Returns:
            Dictionary containing state_mean, state_std
        """
        if not self.is_normalized:
            return {
                "state_mean": None,
                "state_std": None
            }
        
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std
        }


class DynamicsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for dynamics learning.
    Handles data loading, splitting, normalization, and batch preparation.
    """
    
    def __init__(
        self,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
        next_states: torch.Tensor = None,
        state_derivatives: Optional[torch.Tensor] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        random_seed: int = 42,
        data_path: Optional[str] = None,
        persistent_workers: bool = False
    ) -> None:
        """
        Initialize the data module.
        
        Args:
            states: Tensor of states [n_samples, state_dim]
            actions: Tensor of actions [n_samples, action_dim]
            next_states: Tensor of next states [n_samples, state_dim]
            state_derivatives: Optional tensor of state derivatives [n_samples, state_dim]
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            normalize: Whether to normalize the data
            random_seed: Random seed for reproducibility
            data_path: Optional path to load data from (instead of using provided tensors)
            persistent_workers: Whether to keep worker processes alive between batches
        """
        super().__init__()
        
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.state_derivatives = state_derivatives
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.random_seed = random_seed
        self.data_path = data_path
        self.persistent_workers = persistent_workers
        
        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: train_ratio + val_ratio + test_ratio = {total_ratio}, adjusting to 1.0")
            self.train_ratio = train_ratio / total_ratio
            self.val_ratio = val_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
    
    def prepare_data(self) -> None:
        """
        Prepare the data if needed (download, etc.).
        For this module, either data was provided directly or will be loaded from file.
        """
        if self.data_path is not None:
            # Load data from file if provided
            try:
                data = torch.load(self.data_path)
                self.states = data.get("states")
                self.actions = data.get("actions")
                self.next_states = data.get("next_states")
                self.state_derivatives = data.get("state_derivatives")
            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}")
                raise
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation, and testing.
        
        Args:
            stage: Optional stage to setup ('fit', 'validate', 'test')
        """
        # Validate that we have the required data
        if self.states is None or self.actions is None or self.next_states is None:
            raise ValueError("States, actions, and next_states must be provided.")
        
        # Create full dataset
        full_dataset = DynamicsDataset(
            states=self.states,
            actions=self.actions,
            next_states=self.next_states,
            state_derivatives=self.state_derivatives,
            normalize=self.normalize
        )
        
        # Calculate split sizes
        n_samples = len(full_dataset)
        n_train = int(self.train_ratio * n_samples)
        n_val = int(self.val_ratio * n_samples)
        n_test = n_samples - n_train - n_val
        
        # Create splits
        generator = torch.Generator().manual_seed(self.random_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )
        
        # Store normalization stats for potential later use
        self.normalization_stats = full_dataset.get_normalization_stats()
    
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get the normalization statistics.
        
        Returns:
            Dictionary containing normalization statistics
        """
        return self.normalization_stats if hasattr(self, "normalization_stats") else {}


class CLFDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for CLF learning.
    Handles data loading, splitting, normalization, and batch preparation.
    """
    
    def __init__(
        self,
        states: torch.Tensor = None,
        dynamics_model: Optional[torch.nn.Module] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True,
        random_seed: int = 42,
        data_path: Optional[str] = None,
        persistent_workers: bool = False
    ) -> None:
        """
        Initialize the data module.
        
        Args:
            states: Tensor of states [n_samples, state_dim]
            dynamics_model: Optional dynamics model for Lie derivative computation
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            normalize: Whether to normalize the data
            random_seed: Random seed for reproducibility
            data_path: Optional path to load data from (instead of using provided tensors)
            persistent_workers: Whether to keep worker processes alive between batches
        """
        super().__init__()
        
        self.states = states
        self.dynamics_model = dynamics_model
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.random_seed = random_seed
        self.data_path = data_path
        self.persistent_workers = persistent_workers
        
        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: train_ratio + val_ratio + test_ratio = {total_ratio}, adjusting to 1.0")
            self.train_ratio = train_ratio / total_ratio
            self.val_ratio = val_ratio / total_ratio
            self.test_ratio = test_ratio / total_ratio
    
    def prepare_data(self) -> None:
        """
        Prepare the data if needed (download, etc.).
        For this module, either data was provided directly or will be loaded from file.
        """
        if self.data_path is not None:
            # Load data from file if provided
            try:
                data = torch.load(self.data_path)
                self.states = data.get("states")
                
                # Note: dynamics_model should be provided separately
            except Exception as e:
                print(f"Error loading data from {self.data_path}: {e}")
                raise
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation, and testing.
        
        Args:
            stage: Optional stage to setup ('fit', 'validate', 'test')
        """
        # Validate that we have the required data
        if self.states is None:
            raise ValueError("States must be provided.")
        
        # Create full dataset
        full_dataset = CLFDataset(
            states=self.states,
            dynamics_model=self.dynamics_model,
            normalize=self.normalize
        )
        
        # Calculate split sizes
        n_samples = len(full_dataset)
        n_train = int(self.train_ratio * n_samples)
        n_val = int(self.val_ratio * n_samples)
        n_test = n_samples - n_train - n_val
        
        # Create splits
        generator = torch.Generator().manual_seed(self.random_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )
        
        # Store normalization stats for potential later use
        self.normalization_stats = full_dataset.get_normalization_stats()
    
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        persistent_workers = getattr(self, 'persistent_workers', False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers and self.num_workers > 0
        )
    
    def get_normalization_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get the normalization statistics.
        
        Returns:
            Dictionary containing normalization statistics
        """
        return self.normalization_stats if hasattr(self, "normalization_stats") else {}