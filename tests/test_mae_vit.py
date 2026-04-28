from __future__ import annotations

import pytest
import torch

from vision.diff_renderer import DifferentiableShapeRenderer
from vision.mae_vit_model import MAEViTEncoder, MultiScaleViTWakeNet, select_device


class TestMAEViTForward:
    def test_encoder_output_shape(self):
        enc = MAEViTEncoder(in_channels=4, pretrained=False)
        x = torch.randn(2, 4, 128, 128)
        out = enc(x)
        assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"

    def test_encoder_resize_to_224(self):
        enc = MAEViTEncoder(in_channels=4, pretrained=False)
        x = torch.randn(2, 4, 128, 128)
        out1 = enc(x)
        x2 = torch.randn(2, 4, 64, 64)
        out2 = enc(x2)
        assert out1.shape == out2.shape == (2, 768)

    def test_multiscale_vit_output_shapes(self):
        model = MultiScaleViTWakeNet(n_scales=4, in_channels=4, n_shapes=5, n_re_classes=3, pretrained=False)
        x = torch.randn(4, 4, 4, 128, 128)
        out = model(x)
        assert out["shape_logits"].shape == (4, 5)
        assert out["params_pred"].shape == (4, 2)
        assert out["re_logits"].shape == (4, 3)

    def test_multiscale_vit_single_scale(self):
        model = MultiScaleViTWakeNet(n_scales=1, in_channels=4, n_shapes=5, n_re_classes=3, pretrained=False)
        x = torch.randn(4, 1, 4, 128, 128)
        out = model(x)
        assert out["shape_logits"].shape == (4, 5)

    def test_freeze_backbone(self):
        model = MultiScaleViTWakeNet(n_scales=4, in_channels=4, n_shapes=5, n_re_classes=3, pretrained=False)
        assert all(p.requires_grad for p in model.encoder.parameters())
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.encoder.parameters())

    def test_unfreeze_with_llrd(self):
        model = MultiScaleViTWakeNet(n_scales=4, in_channels=4, n_shapes=5, n_re_classes=3, pretrained=False)
        groups = model.unfreeze_with_llrd(base_lr=1e-4, llrd_decay=0.85)
        lrs = [g["lr"] for g in groups if g["params"]]
        assert len(lrs) > 0
        assert all(lr > 0 for lr in lrs)


class TestDiffRenderer:
    def test_circle_render_produces_valid_mask(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=2)
        renderer.eval()
        shape_idx = torch.tensor([0])
        dy = torch.tensor([0.0])
        eps = torch.tensor([0.0])
        loss, pred, target = renderer.render_and_loss(shape_idx, dy, eps, shape_idx, dy, eps)
        assert pred.shape == target.shape == (1, 1, 128, 128)
        assert 0.0 <= float(pred.min()) <= float(pred.max()) <= 1.0
        assert float(pred.max()) > 0.9, f"Circle mask should reach near 1.0, got {float(pred.max())}"
        assert float(loss) < 0.5, f"Loss = 1-Dice, expect ~0.27, got {float(loss)}"

    def test_airfoil_render_produces_valid_mask(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=2)
        renderer.eval()
        shape_idx = torch.tensor([2])
        dy = torch.tensor([0.0])
        eps = torch.tensor([0.0])
        loss, pred, target = renderer.render_and_loss(shape_idx, dy, eps, shape_idx, dy, eps)
        assert 0.0 <= float(pred.min()) <= float(pred.max()) <= 1.0
        assert float(pred.max()) > 0.8, f"Airfoil mask should reach near 1.0, got {float(pred.max())}"

    def test_all_five_shapes_render(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=1)
        shape_idx = torch.tensor([0, 1, 2, 3, 4])
        dy = torch.zeros(5)
        eps = torch.zeros(5)
        masks = renderer.render(shape_idx, dy, eps)
        assert masks.shape == (5, 1, 128, 128)
        assert 0.0 <= float(masks.min()) <= float(masks.max()) <= 1.0
        for i in range(5):
            assert float(masks[i].max()) > 0.5, f"Shape {i} mask should reach >0.5 (got max={float(masks[i].max()):.4f})"

    def test_different_shapes_produce_different_masks(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=1)
        m0 = renderer.render(torch.tensor([0]), torch.zeros(1), torch.zeros(1))
        m2 = renderer.render(torch.tensor([2]), torch.zeros(1), torch.zeros(1))
        diff = float((m0 - m2).abs().mean())
        assert diff > 0.01, f"Circle vs airfoil should differ significantly (diff={diff:.6f})"

    def test_dy_shift_changes_mask(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=2)
        m0 = renderer.render(torch.tensor([0]), torch.tensor([0.0]), torch.zeros(1))
        m1 = renderer.render(torch.tensor([0]), torch.tensor([0.1]), torch.zeros(1))
        diff = float((m0 - m1).abs().mean())
        assert diff > 0.001, f"dy shift should change mask (diff={diff})"

    def test_eps_deformation_changes_mask(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=2)
        m0 = renderer.render(torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]))
        m1 = renderer.render(torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.05]))
        diff = float((m0 - m1).abs().mean())
        assert diff > 0.001, f"eps deformation should change full-geometry mask (diff={diff})"

    def test_backward_pass_creates_gradients(self):
        renderer = DifferentiableShapeRenderer(image_size=128, transition_px=2)
        renderer.train()
        shape_idx = torch.tensor([0, 0])
        dy_pred = torch.tensor([0.0, 0.1], requires_grad=True)
        dy_target = torch.tensor([0.0, 0.0])
        eps = torch.zeros(2)
        loss, pred_mask, _ = renderer.render_and_loss(
            shape_idx, dy_pred, eps, shape_idx, dy_target, eps
        )
        assert pred_mask.shape == (2, 1, 128, 128)
        assert float(loss.detach()) > 0.001, "Mismatched params should produce non-zero loss"
        loss.backward()
        assert dy_pred.grad is not None and dy_pred.grad.abs().sum() > 0, "dy gradient should be non-zero when masks differ"

    def test_eval_mode_produces_sharp_masks(self):
        renderer = DifferentiableShapeRenderer(image_size=64, transition_px=1)
        renderer.eval()
        shape_idx = torch.tensor([0])
        mask = renderer.render(shape_idx, torch.zeros(1), torch.zeros(1))
        assert float(mask.max()) > 0.9, f"Circle mask should be nearly binary, got max={float(mask.max())}"

    def test_soft_dice_loss_bounded(self):
        renderer = DifferentiableShapeRenderer(image_size=32, transition_px=2)
        renderer.eval()
        pred = torch.rand(2, 1, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        loss = renderer.soft_dice_loss(pred, target)
        assert 0.0 <= float(loss) <= 2.0, f"Dice loss should be in [0,2], got {float(loss)}"

    def test_render_and_loss_forward_backward(self):
        renderer = DifferentiableShapeRenderer(image_size=64, transition_px=2)
        renderer.train()
        shape_idx = torch.tensor([0])
        dy = torch.tensor([0.05], requires_grad=True)
        eps = torch.zeros(1)
        loss, pred, target = renderer.render_and_loss(shape_idx, dy, eps, shape_idx, dy, eps)
        assert pred.shape == target.shape == (1, 1, 64, 64)
        loss.backward()
        assert dy.grad is not None and dy.grad.abs().sum() > 0, "dy gradient should be non-zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
