
import time
import numpy as np
import torch

from src.horizon_utils import build_supervised
from src.horizon_metrics import (
    evaluate_mse,
    rolling_rmse,
    horizon_from_rmse
)
from src.horizon_models import LinearAR, TorchSeqWrapper, TorchWrapper
from src.horizon_progress import ProgressBar
from src.horizon_training import (
    build_multistep_supervised,
    train_lstm,
    train_lstm_multistep,
    train_mlp,
    train_mlp_multistep,
)

class Forecaster:
    """Handles model training and forecasting (embedding selection, final model)."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.best_config = None
        self.model = None
        self.search_history = []

    def select_embedding(self, train_series, val_series):
        """Selects the best (dim, lag) embedding based on validation criteria."""
        dim_values = list(range(self.args.dim_min, self.args.dim_max + 1))
        lag_values = list(range(self.args.lag_min, self.args.lag_max + 1))
        
        best = None
        progress = None
        if self.args.progress:
            progress = ProgressBar(len(dim_values) * len(lag_values), label="embed-search")
            
        for dim in dim_values:
            for lag in lag_values:
                try:
                    if self.args.train_multistep and self.args.model in ("mlp", "lstm"):
                        x_train, y_train = build_multistep_supervised(
                            train_series, dim, lag, horizon=self.args.train_horizon
                        )
                        x_val, y_val = build_multistep_supervised(
                            val_series, dim, lag, horizon=self.args.train_horizon
                        )
                    else:
                        x_train, y_train = build_supervised(
                            train_series, dim, lag, horizon=1
                        )
                        x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
                except ValueError:
                    if progress:
                        progress.update(1, extra=f"dim={dim} lag={lag}")
                    continue

                if self.args.model == "linear":
                    model = LinearAR(reg=self.args.linear_reg).fit(x_train, y_train)
                    val_loss = evaluate_mse(model, x_val, y_val)
                    wrapped = model
                elif self.args.model == "mlp":
                    if self.args.train_multistep:
                        model, val_loss = train_mlp_multistep(
                            x_train, y_train, x_val, y_val,
                            input_dim=dim,
                            hidden_dim=self.args.mlp_hidden,
                            epochs=self.args.mlp_epochs,
                            lr=self.args.mlp_lr,
                            batch_size=self.args.mlp_batch,
                            patience=self.args.mlp_patience,
                            tf_start=self.args.tf_start,
                            tf_end=self.args.tf_end,
                            tf_val=self.args.tf_val,
                            device=self.device,
                            show_progress=False,
                        )
                    else:
                        model, val_loss = train_mlp(
                            x_train, y_train, x_val, y_val,
                            input_dim=dim,
                            hidden_dim=self.args.mlp_hidden,
                            epochs=self.args.mlp_epochs,
                            lr=self.args.mlp_lr,
                            batch_size=self.args.mlp_batch,
                            patience=self.args.mlp_patience,
                            device=self.device,
                            show_progress=False,
                        )
                    wrapped = TorchWrapper(model, self.device)
                else: # LSTM
                    if self.args.train_multistep:
                        model, val_loss = train_lstm_multistep(
                            x_train, y_train, x_val, y_val,
                            hidden_dim=self.args.lstm_hidden,
                            num_layers=self.args.lstm_layers,
                            epochs=self.args.lstm_epochs,
                            lr=self.args.lstm_lr,
                            batch_size=self.args.lstm_batch,
                            patience=self.args.lstm_patience,
                            tf_start=self.args.tf_start,
                            tf_end=self.args.tf_end,
                            tf_val=self.args.tf_val,
                            device=self.device,
                            show_progress=False,
                        )
                    else:
                        model, val_loss = train_lstm(
                            x_train, y_train, x_val, y_val,
                            hidden_dim=self.args.lstm_hidden,
                            num_layers=self.args.lstm_layers,
                            epochs=self.args.lstm_epochs,
                            lr=self.args.lstm_lr,
                            batch_size=self.args.lstm_batch,
                            patience=self.args.lstm_patience,
                            device=self.device,
                            show_progress=False,
                        )
                    wrapped = TorchSeqWrapper(model, self.device)

                selection = {
                    "metric": self.args.selection_metric,
                    "score": -val_loss,
                    "horizon": None,
                }
                if self.args.selection_metric == "horizon":
                    rmse_val = rolling_rmse(
                        wrapped, val_series, dim, lag, self.args.selection_horizon_max
                    )
                    base_err = rmse_val[0] if rmse_val.size > 0 else 0.0
                    if self.args.error_mode == "relative":
                        tolerance = base_err * self.args.error_factor
                    else:
                        tolerance = self.args.error_tolerance
                    
                    if not np.isfinite(tolerance) or tolerance <= 0:
                        horizon_val = 0
                    else:
                        horizon_val = horizon_from_rmse(rmse_val, tolerance)
                    selection["score"] = horizon_val
                    selection["horizon"] = horizon_val

                self.search_history.append(
                    {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": float(val_loss),
                        "selection_metric": self.args.selection_metric,
                        "selection_score": float(selection["score"]),
                        "selection_horizon": selection["horizon"],
                    }
                )

                if best is None:
                    best = {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": val_loss,
                        "model": wrapped,
                        "selection": selection,
                    }
                else:
                    score = selection["score"]
                    best_score = best["selection"]["score"]
                    if score > best_score:
                        is_better = True
                    elif score == best_score:
                         # tie-break with val_loss
                         is_better = val_loss < best["val_loss"]
                    else:
                        is_better = False
                    
                    if is_better:
                         best = {
                            "dim": dim,
                            "lag": lag,
                            "val_loss": val_loss,
                            "model": wrapped,
                            "selection": selection,
                        }

                if progress:
                    extra = f"dim={dim} lag={lag} val={val_loss:.4f}"
                    if selection["horizon"] is not None:
                        extra += f" h={selection['horizon']}"
                    progress.update(1, extra=extra)

        if best is None:
            raise RuntimeError("No valid embedding configuration found.")
        if progress:
            progress.close()
            
        self.best_config = best
        best["search_history"] = list(self.search_history)
        return best

    def train_final_model(self, train_series, val_series):
        """Trains the final model on train+val with chosen embedding."""
        if self.best_config is None:
             raise RuntimeError("Must call select_embedding first.")
        
        dim = self.best_config["dim"]
        lag = self.best_config["lag"]
        merged = np.concatenate([train_series, val_series], axis=0)
        
        if self.args.train_multistep and self.args.model in ("mlp", "lstm"):
            x_train, y_train = build_multistep_supervised(
                merged, dim, lag, horizon=self.args.train_horizon
            )
            x_val, y_val = build_multistep_supervised(
                val_series, dim, lag, horizon=self.args.train_horizon
            )
        else:
            x_train, y_train = build_supervised(merged, dim, lag, horizon=1)
            x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
            
        if self.args.model == "linear":
            model = LinearAR(reg=self.args.linear_reg).fit(x_train, y_train)
            self.model = model
            return model
            
        if self.args.model == "mlp":
            if self.args.train_multistep:
               model, _ = train_mlp_multistep(
                   x_train, y_train, x_val, y_val,
                   input_dim=dim,
                   hidden_dim=self.args.mlp_hidden,
                   epochs=self.args.mlp_epochs,
                   lr=self.args.mlp_lr,
                   batch_size=self.args.mlp_batch,
                   patience=self.args.mlp_patience,
                   tf_start=self.args.tf_start,
                   tf_end=self.args.tf_end,
                   tf_val=self.args.tf_val,
                   device=self.device,
                   show_progress=self.args.progress,
               )
            else:
               model, _ = train_mlp(
                   x_train, y_train, x_val, y_val,
                   input_dim=dim,
                   hidden_dim=self.args.mlp_hidden,
                   epochs=self.args.mlp_epochs,
                   lr=self.args.mlp_lr,
                   batch_size=self.args.mlp_batch,
                   patience=self.args.mlp_patience,
                   device=self.device,
                   show_progress=self.args.progress,
               )
            self.model = TorchWrapper(model, self.device)
            return self.model

        # LSTM
        if self.args.train_multistep:
            model, _ = train_lstm_multistep(
                x_train, y_train, x_val, y_val,
                hidden_dim=self.args.lstm_hidden,
                num_layers=self.args.lstm_layers,
                epochs=self.args.lstm_epochs,
                lr=self.args.lstm_lr,
                batch_size=self.args.lstm_batch,
                patience=self.args.lstm_patience,
                tf_start=self.args.tf_start,
                tf_end=self.args.tf_end,
                tf_val=self.args.tf_val,
                device=self.device,
                show_progress=self.args.progress,
            )
        else:
            model, _ = train_lstm(
                x_train, y_train, x_val, y_val,
                hidden_dim=self.args.lstm_hidden,
                num_layers=self.args.lstm_layers,
                epochs=self.args.lstm_epochs,
                lr=self.args.lstm_lr,
                batch_size=self.args.lstm_batch,
                patience=self.args.lstm_patience,
                device=self.device,
                show_progress=self.args.progress,
            )
        self.model = TorchSeqWrapper(model, self.device)
        return self.model
