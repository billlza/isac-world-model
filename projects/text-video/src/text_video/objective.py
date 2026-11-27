"""Flow and temporal supervision; confidence is supplied by the dataset only."""

import torch


def flow_objective(predicted_velocity, clean_video, noise, time, reference_weight, temporal_weight=0.1, *, flow_scale=1.0, temporal_scale=1.0):
    if predicted_velocity.shape != clean_video.shape or noise.shape != clean_video.shape or clean_video.ndim != 5:
        raise ValueError("Expected matching [B,C,T,H,W] video, noise, and velocity")
    if time.shape != (clean_video.shape[0],) or reference_weight.shape != (clean_video.shape[0], clean_video.shape[2]):
        raise ValueError("Expected time [B] and externally supplied reference_weight [B,T]")
    if not torch.isfinite(reference_weight).all() or (reference_weight < 0).any() or (reference_weight > 1).any():
        raise ValueError("reference_weight must be finite and in [0,1]")
    if not torch.isfinite(time).all() or (time < 0).any() or (time > 1).any():
        raise ValueError("time must be finite and in [0,1]")
    # Float32 reductions are intentional, even when attention uses bfloat16.
    residual = predicted_velocity.float() - (clean_video.float() - noise.float())
    per_frame = residual.square().mean(dim=(1, 3, 4))
    weights = reference_weight.float().detach()
    support = weights.sum()
    if support.item() <= 0:
        raise ValueError("Batch has no trusted reference support")
    flow_loss = (per_frame * weights).sum() / support
    # x1_hat - x1 = (1-t) * velocity_error; matching target deltas does not
    # reward static clips or feed future clean frames into the generation input.
    reconstruction_error = (1 - time.float()).view(-1, 1, 1, 1, 1) * residual
    if clean_video.shape[2] > 1:
        delta_error = reconstruction_error[:, :, 1:] - reconstruction_error[:, :, :-1]
        pair_weights = torch.minimum(weights[:, 1:], weights[:, :-1])
        pair_support = pair_weights.sum()
        pair_mse = delta_error.square().mean(dim=(1, 3, 4))
        temporal_loss = (pair_mse * pair_weights).sum() / pair_support.clamp_min(1e-12)
    else:
        pair_support = support.new_zeros(())
        temporal_loss = residual.sum() * 0
    total = flow_scale * flow_loss + temporal_scale * temporal_weight * temporal_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training objective")
    metrics = {
        "flow_loss": float(flow_loss.detach()), "temporal_loss": float(temporal_loss.detach()),
        "total_loss": float((flow_loss + temporal_weight * temporal_loss).detach()), "reference_support": float(support),
        "temporal_support": float(pair_support),
        "frame_mse_unweighted": float(per_frame.detach().mean()),
    }
    return total, metrics
