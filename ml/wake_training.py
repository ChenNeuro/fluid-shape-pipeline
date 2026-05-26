from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vision.wake_model import MultiScaleWakeNet, compute_multitask_loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_loader(
    x: np.ndarray,
    shape_idx: np.ndarray,
    params: np.ndarray,
    re_idx: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(shape_idx).long(),
        torch.from_numpy(params).float(),
        torch.from_numpy(re_idx).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def augment_batch(x: torch.Tensor, aug_cfg: dict) -> torch.Tensor:
    if not aug_cfg.get("enabled", False):
        return x
    flip_prob = float(aug_cfg.get("random_vertical_flip", 0.0))
    if flip_prob > 0:
        raise ValueError(
            "random_vertical_flip is disabled: wake labels and vector channels are not "
            "invariant under a vertical reflection."
        )
    noise_std = float(aug_cfg.get("random_noise_std", 0.0))
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


def predict_wake_model(
    model: torch.nn.Module, x: np.ndarray, *, batch_size: int, device: torch.device
) -> dict[str, np.ndarray]:
    dataset = TensorDataset(torch.from_numpy(x).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    shape_logits = []
    params_pred = []
    re_logits = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            outputs = model(batch_x.to(device))
            shape_logits.append(outputs["shape_logits"].detach().cpu().numpy())
            params_pred.append(outputs["params_pred"].detach().cpu().numpy())
            re_logits.append(outputs["re_logits"].detach().cpu().numpy())

    return {
        "shape_logits": np.concatenate(shape_logits, axis=0),
        "params_pred": np.concatenate(params_pred, axis=0),
        "re_logits": np.concatenate(re_logits, axis=0),
    }


def _run_supervised_epochs(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    loss_weights: dict,
    aug_cfg: dict,
    patience: int,
    history: list[dict[str, float]] | None = None,
    phase: str | int | None = None,
    epoch_offset: int = 0,
    max_grad_norm: float | None = None,
) -> list[dict[str, float]]:
    history = [] if history is None else history
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        loss_shape = 0.0
        loss_params = 0.0
        loss_re = 0.0
        n_items = 0

        for batch_x, batch_shape, batch_params, batch_re in train_loader:
            batch_x = augment_batch(batch_x.to(device), aug_cfg)
            batch_shape = batch_shape.to(device)
            batch_params = batch_params.to(device)
            batch_re = batch_re.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_x)
            loss, loss_parts = compute_multitask_loss(
                outputs,
                shape_target=batch_shape,
                param_target=batch_params,
                re_target=batch_re,
                loss_weights=loss_weights,
            )
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_n = int(batch_x.shape[0])
            n_items += batch_n
            loss_total += loss_parts["loss_total"] * batch_n
            loss_shape += loss_parts["loss_shape"] * batch_n
            loss_params += loss_parts["loss_params"] * batch_n
            loss_re += loss_parts["loss_re"] * batch_n

        epoch_entry: dict[str, float] = {
            "epoch": epoch_offset + epoch + 1,
            "loss_total": loss_total / max(n_items, 1),
            "loss_shape": loss_shape / max(n_items, 1),
            "loss_params": loss_params / max(n_items, 1),
            "loss_re": loss_re / max(n_items, 1),
        }
        if phase is not None:
            epoch_entry["phase"] = phase

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_x, batch_shape, batch_params, batch_re in val_loader:
                    batch_x = batch_x.to(device)
                    batch_shape = batch_shape.to(device)
                    batch_params = batch_params.to(device)
                    batch_re = batch_re.to(device)
                    outputs = model(batch_x)
                    loss, _ = compute_multitask_loss(
                        outputs,
                        shape_target=batch_shape,
                        param_target=batch_params,
                        re_target=batch_re,
                        loss_weights=loss_weights,
                    )
                    batch_n = int(batch_x.shape[0])
                    val_n += batch_n
                    val_loss_total += float(loss.detach().cpu()) * batch_n
            val_loss = val_loss_total / max(val_n, 1)
            epoch_entry["val_loss"] = val_loss

            if patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        history.append(epoch_entry)
                        break

        history.append(epoch_entry)
        if scheduler is not None:
            scheduler.step()

    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
    return history


def _supervised_loaders(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    train_loader = tensor_loader(
        x_train, shape_train_idx, params_train, re_train_idx, batch_size=batch_size, shuffle=True
    )
    val_loader = None
    if x_val.shape[0] > 0:
        val_loader = tensor_loader(
            x_val, shape_val_idx, params_val, re_val_idx, batch_size=batch_size, shuffle=False
        )
    return train_loader, val_loader


def train_resnet_wake_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[MultiScaleWakeNet, list[dict[str, float]]]:
    train_cfg = cfg.get("vision", {}).get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 8))
    scheduler_cfg = train_cfg.get("scheduler", {})

    set_seed(seed)
    model = MultiScaleWakeNet(
        n_scales=int(x_train.shape[1]),
        in_channels=int(x_train.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        fusion_hidden=int(train_cfg.get("fusion_hidden", 256)),
        dropout=float(train_cfg.get("dropout", 0.15)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(scheduler_cfg.get("type", "none")) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(scheduler_cfg.get("T_max", epochs))
        )
    train_loader, val_loader = _supervised_loaders(
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        batch_size=batch_size,
    )
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        loss_weights=train_cfg.get("loss_weights", {}),
        aug_cfg=train_cfg.get("augmentation", {}),
        patience=int(train_cfg.get("early_stopping_patience", 0)),
    )
    return model, history


def train_vit_wake_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from vision.mae_vit_model import MultiScaleViTWakeNet

    vit_cfg = cfg.get("vision", {}).get("mae_vit", {})
    batch_size = int(vit_cfg.get("batch_size", 8))
    phase1_epochs = int(vit_cfg.get("phase1_epochs", 12))
    phase2_epochs = int(vit_cfg.get("phase2_epochs", 13))
    phase1_lr = float(vit_cfg.get("phase1_lr", 2e-3))
    phase2_base_lr = float(vit_cfg.get("phase2_base_lr", 5e-5))
    llrd_decay = float(vit_cfg.get("llrd_decay", 0.85))
    loss_weights = vit_cfg.get(
        "loss_weights", cfg.get("vision", {}).get("training", {}).get("loss_weights", {})
    )
    patience = int(vit_cfg.get("early_stopping_patience", 0))
    aug_cfg = cfg.get("vision", {}).get("training", {}).get("augmentation", {})

    set_seed(seed)
    model = MultiScaleViTWakeNet(
        n_scales=int(x_train.shape[1]),
        in_channels=int(x_train.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        proj_dim=int(vit_cfg.get("proj_dim", 512)),
        fusion_hidden=int(vit_cfg.get("fusion_hidden", 512)),
        dropout=float(vit_cfg.get("dropout", 0.2)),
        pretrained=True,
    ).to(device)
    train_loader, val_loader = _supervised_loaders(
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        batch_size=batch_size,
    )

    history: list[dict[str, float]] = []
    model.freeze_backbone()
    optimizer_p1 = torch.optim.AdamW(
        list(model.scale_proj.parameters())
        + list(model.fusion.parameters())
        + list(model.shape_head.parameters())
        + list(model.params_head.parameters())
        + list(model.re_head.parameters()),
        lr=phase1_lr,
        weight_decay=1e-4,
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs)
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        epochs=phase1_epochs,
        device=device,
        loss_weights=loss_weights,
        aug_cfg=aug_cfg,
        patience=patience,
        history=history,
        phase=1,
    )

    llrd_groups = model.unfreeze_with_llrd(base_lr=phase2_base_lr, llrd_decay=llrd_decay)
    optimizer_p2 = torch.optim.AdamW(llrd_groups, lr=phase2_base_lr)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=phase2_epochs)
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        epochs=phase2_epochs,
        device=device,
        loss_weights=loss_weights,
        aug_cfg=aug_cfg,
        patience=patience,
        history=history,
        phase=2,
        max_grad_norm=1.0,
    )
    return model, history


def train_jepa_wake_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from vision.jepa_encoder import IJEPAPretrainer, LightweightCNNEncoder
    from vision.wake_model import MultiScaleJEPAModel

    jepa_cfg = cfg.get("vision", {}).get("jepa", {})
    batch_size = int(jepa_cfg.get("batch_size", 16))
    pretrain_epochs = int(jepa_cfg.get("pretrain_epochs", 30))
    feature_dim = int(jepa_cfg.get("feature_dim", 192))
    ft_epochs = int(jepa_cfg.get("fine_tune_epochs", 30))
    scheduler_cfg = cfg.get("vision", {}).get("training", {}).get("scheduler", {})

    set_seed(seed)
    in_channels = int(x_train.shape[2])
    n_scales = x_train.shape[1]
    crops_for_jepa = x_train.transpose(1, 0, 2, 3, 4).reshape(
        -1, in_channels, x_train.shape[3], x_train.shape[4]
    )
    crops_for_jepa = np.asarray(crops_for_jepa, dtype=np.float32)

    jepa_dataset = TensorDataset(torch.from_numpy(crops_for_jepa).float())
    jepa_loader = DataLoader(jepa_dataset, batch_size=batch_size, shuffle=True)
    encoder = LightweightCNNEncoder(in_channels=in_channels, feature_dim=feature_dim).to(device)
    pretrainer = IJEPAPretrainer(
        encoder,
        feature_dim=feature_dim,
        proj_dim=int(jepa_cfg.get("proj_dim", 128)),
        mask_ratio=float(jepa_cfg.get("mask_ratio", 0.3)),
        block_size=int(jepa_cfg.get("block_size", 8)),
        momentum=float(jepa_cfg.get("momentum", 0.996)),
    ).to(device)
    pretrain_opt = torch.optim.AdamW(
        pretrainer.context_encoder.parameters(),
        lr=float(jepa_cfg.get("lr", 0.001)),
    )
    for param in pretrainer.predictor.parameters():
        pretrain_opt.add_param_group({"params": param})

    history: list[dict[str, float]] = []
    for epoch in range(pretrain_epochs):
        pretrainer.train()
        total_loss = 0.0
        n = 0
        for (batch_x,) in jepa_loader:
            batch_x = batch_x.to(device)
            pretrain_opt.zero_grad(set_to_none=True)
            loss = pretrainer(batch_x)
            loss.backward()
            pretrain_opt.step()
            bn = int(batch_x.shape[0])
            n += bn
            total_loss += float(loss.detach().cpu()) * bn
        history.append(
            {
                "epoch": epoch + 1,
                "phase": "pretrain",
                "loss_total": total_loss / max(n, 1),
            }
        )

    jepa_encoder = pretrainer.extract_encoder().cpu()
    del pretrainer, pretrain_opt, jepa_loader, jepa_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = MultiScaleJEPAModel(
        n_scales=n_scales,
        in_channels=in_channels,
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        feature_dim=feature_dim,
        fusion_hidden=int(jepa_cfg.get("fusion_hidden", 192)),
        dropout=float(jepa_cfg.get("dropout", 0.15)),
        pretrained_encoder=jepa_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(jepa_cfg.get("fine_tune_lr", 0.0005)), weight_decay=1e-4
    )
    scheduler = None
    if str(scheduler_cfg.get("type", "none")) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(scheduler_cfg.get("T_max", ft_epochs))
        )
    train_loader, val_loader = _supervised_loaders(
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        batch_size=batch_size,
    )
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=ft_epochs,
        device=device,
        loss_weights=jepa_cfg.get(
            "loss_weights", cfg.get("vision", {}).get("training", {}).get("loss_weights", {})
        ),
        aug_cfg=cfg.get("vision", {}).get("training", {}).get("augmentation", {}),
        patience=int(jepa_cfg.get("early_stopping_patience", 10)),
        history=history,
        phase="finetune",
        epoch_offset=pretrain_epochs,
    )
    return model, history


def train_smallcnn_wake_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from vision.wake_model import MultiScaleJEPAModel

    train_cfg = cfg.get("vision", {}).get("training", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 80))
    scheduler_cfg = train_cfg.get("scheduler", {})

    set_seed(seed)
    model = MultiScaleJEPAModel(
        n_scales=int(x_train.shape[1]),
        in_channels=int(x_train.shape[2]),
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        feature_dim=int(cfg.get("vision", {}).get("jepa", {}).get("feature_dim", 192)),
        fusion_hidden=int(train_cfg.get("fusion_hidden", 192)),
        dropout=float(train_cfg.get("dropout", 0.15)),
        pretrained_encoder=None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.0008)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(scheduler_cfg.get("type", "none")) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(scheduler_cfg.get("T_max", epochs))
        )
    train_loader, val_loader = _supervised_loaders(
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        batch_size=batch_size,
    )
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        loss_weights=train_cfg.get("loss_weights", {}),
        aug_cfg=train_cfg.get("augmentation", {}),
        patience=int(train_cfg.get("early_stopping_patience", 10)),
    )
    return model, history


def train_simsiam_wake_model(
    *,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    from vision.jepa_encoder import LightweightCNNEncoder
    from vision.simsiam_pretrainer import SimSiamPretrainer
    from vision.wake_model import MultiScaleJEPAModel

    jepa_cfg = cfg.get("vision", {}).get("jepa", {})
    batch_size = int(jepa_cfg.get("batch_size", 16))
    pretrain_epochs = int(jepa_cfg.get("pretrain_epochs", 30))
    feature_dim = int(jepa_cfg.get("feature_dim", 192))
    ft_epochs = int(jepa_cfg.get("fine_tune_epochs", 30))
    scheduler_cfg = cfg.get("vision", {}).get("training", {}).get("scheduler", {})

    set_seed(seed)
    in_channels = int(x_train.shape[2])
    n_scales = x_train.shape[1]
    crops = x_train.transpose(1, 0, 2, 3, 4).reshape(
        -1, in_channels, x_train.shape[3], x_train.shape[4]
    )
    crops = np.asarray(crops, dtype=np.float32)
    ssl_dataset = TensorDataset(torch.from_numpy(crops).float())
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True)

    encoder = LightweightCNNEncoder(in_channels=in_channels, feature_dim=feature_dim).to(device)
    ssl_trainer = SimSiamPretrainer(
        encoder, feature_dim=feature_dim, noise_std=0.02, flip_prob=0.5
    ).to(device)
    ssl_opt = torch.optim.AdamW(
        list(encoder.parameters())
        + list(ssl_trainer.projector.parameters())
        + list(ssl_trainer.predictor.parameters()),
        lr=float(jepa_cfg.get("lr", 0.001)),
    )

    history: list[dict[str, float]] = []
    for epoch in range(pretrain_epochs):
        ssl_trainer.train()
        total_loss = 0.0
        n = 0
        for (batch_x,) in ssl_loader:
            batch_x = batch_x.to(device)
            ssl_opt.zero_grad(set_to_none=True)
            loss = ssl_trainer(batch_x)
            loss.backward()
            ssl_opt.step()
            bn = int(batch_x.shape[0])
            n += bn
            total_loss += float(loss.detach().cpu()) * bn
        history.append(
            {
                "epoch": epoch + 1,
                "phase": "pretrain_simsiam",
                "loss_total": total_loss / max(n, 1),
            }
        )

    ssl_encoder = ssl_trainer.extract_encoder().cpu()
    del ssl_trainer, ssl_opt, ssl_loader, ssl_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = MultiScaleJEPAModel(
        n_scales=n_scales,
        in_channels=in_channels,
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        feature_dim=feature_dim,
        fusion_hidden=int(jepa_cfg.get("fusion_hidden", 192)),
        dropout=float(jepa_cfg.get("dropout", 0.15)),
        pretrained_encoder=ssl_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(jepa_cfg.get("fine_tune_lr", 0.0005)), weight_decay=1e-4
    )
    scheduler = None
    if str(scheduler_cfg.get("type", "none")) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(scheduler_cfg.get("T_max", ft_epochs))
        )
    train_loader, val_loader = _supervised_loaders(
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        batch_size=batch_size,
    )
    history = _run_supervised_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=ft_epochs,
        device=device,
        loss_weights=jepa_cfg.get(
            "loss_weights", cfg.get("vision", {}).get("training", {}).get("loss_weights", {})
        ),
        aug_cfg=cfg.get("vision", {}).get("training", {}).get("augmentation", {}),
        patience=int(jepa_cfg.get("early_stopping_patience", 10)),
        history=history,
        phase="finetune",
        epoch_offset=pretrain_epochs,
    )
    return model, history


def train_wake_backbone(
    *,
    backbone: str,
    x_train: np.ndarray,
    shape_train_idx: np.ndarray,
    params_train: np.ndarray,
    re_train_idx: np.ndarray,
    x_val: np.ndarray,
    shape_val_idx: np.ndarray,
    params_val: np.ndarray,
    re_val_idx: np.ndarray,
    cfg: dict,
    seed: int,
    n_shapes: int,
    n_re_classes: int,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    trainers = {
        "resnet18": train_resnet_wake_model,
        "mae_vit": train_vit_wake_model,
        "jepa": train_jepa_wake_model,
        "smallcnn": train_smallcnn_wake_model,
        "simsiam": train_simsiam_wake_model,
    }
    if backbone not in trainers:
        raise ValueError(f"Unsupported wake backbone: {backbone}")
    return trainers[backbone](
        x_train=x_train,
        shape_train_idx=shape_train_idx,
        params_train=params_train,
        re_train_idx=re_train_idx,
        x_val=x_val,
        shape_val_idx=shape_val_idx,
        params_val=params_val,
        re_val_idx=re_val_idx,
        cfg=cfg,
        seed=seed,
        n_shapes=n_shapes,
        n_re_classes=n_re_classes,
        device=device,
    )


def save_model_pack(
    *,
    output_path: Path,
    model: torch.nn.Module,
    model_type: str,
    variant_name: str,
    x_shape: tuple[int, ...],
    shape_labels: list[str],
    re_values: list[int],
    test_case_ids: list[str],
    cfg: dict,
    seed: int,
) -> None:
    if model_type == "resnet18":
        model_kwargs = {
            "n_scales": int(x_shape[1]),
            "in_channels": int(x_shape[2]),
            "n_shapes": len(shape_labels),
            "n_re_classes": len(re_values),
            "fusion_hidden": int(
                getattr(
                    model,
                    "fusion_hidden",
                    cfg.get("vision", {}).get("training", {}).get("fusion_hidden", 256),
                )
            ),
            "dropout": float(
                getattr(
                    model,
                    "dropout",
                    cfg.get("vision", {}).get("training", {}).get("dropout", 0.15),
                )
            ),
        }
    elif model_type in ("jepa", "smallcnn", "simsiam"):
        model_kwargs = {
            "n_scales": int(x_shape[1]),
            "in_channels": int(x_shape[2]),
            "n_shapes": len(shape_labels),
            "n_re_classes": len(re_values),
            "feature_dim": int(
                getattr(
                    model,
                    "feature_dim",
                    cfg.get("vision", {}).get("jepa", {}).get("feature_dim", 192),
                )
            ),
            "fusion_hidden": int(
                getattr(
                    model,
                    "fusion_hidden",
                    cfg.get("vision", {}).get("jepa", {}).get("fusion_hidden", 192),
                )
            ),
            "dropout": float(
                getattr(
                    model,
                    "dropout",
                    cfg.get("vision", {}).get("jepa", {}).get("dropout", 0.15),
                )
            ),
        }
    else:
        model_kwargs = {
            "n_scales": int(x_shape[1]),
            "in_channels": int(x_shape[2]),
            "n_shapes": len(shape_labels),
            "n_re_classes": len(re_values),
            "proj_dim": int(cfg.get("vision", {}).get("mae_vit", {}).get("proj_dim", 512)),
            "fusion_hidden": int(
                cfg.get("vision", {}).get("mae_vit", {}).get("fusion_hidden", 512)
            ),
            "dropout": float(cfg.get("vision", {}).get("mae_vit", {}).get("dropout", 0.2)),
        }
    payload = {
        "model_type": model_type,
        "variant_name": variant_name,
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "shape_labels": shape_labels,
        "re_values": re_values,
        "test_case_ids": test_case_ids,
        "fit_seed": int(seed),
        "config_snapshot": cfg,
    }
    torch.save(payload, output_path)
