import torch.nn.functional as F

def multitask_loss(outputs, keypoints, mask, labels):
    classification_loss = F.cross_entropy(outputs["classes"], labels)

    # Each visible joint contributes two coordinates
    valid = mask.unsqueeze(-1).expand_as(keypoints)

    pose_errors = F.smooth_l1_loss(outputs["keypoints"], keypoints, reduction="none")

    pose_loss = ((pose_errors * valid).sum() / valid.sum().clamp(min=1))

    total_loss = classification_loss + 5.0 * pose_loss

    return total_loss, classification_loss, pose_loss