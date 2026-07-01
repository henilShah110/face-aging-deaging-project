import os
import torch

# used for resuming the training loop from exact epoch 
# it was last saved
def save_checkpoint(
    path,
    epoch,
    G_YO,
    G_OY,
    D_old,
    D_young,
    optimizer_G,
    optimizer_D_old,
    optimizer_D_young
):

    torch.save({

        "epoch": epoch,

        "G_YO": G_YO.state_dict(),
        "G_OY": G_OY.state_dict(),

        "D_old": D_old.state_dict(),
        "D_young": D_young.state_dict(),

        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D_old": optimizer_D_old.state_dict(),
        "optimizer_D_young": optimizer_D_young.state_dict()

    }, path)


def load_checkpoint(
    path,
    G_YO,
    G_OY,
    D_old,
    D_young,
    optimizer_G,
    optimizer_D_old,
    optimizer_D_young,
    device
):

    if not os.path.exists(path):

        return 0

    checkpoint = torch.load(
        path,
        map_location=device
    )

    G_YO.load_state_dict(
        checkpoint["G_YO"]
    )

    G_OY.load_state_dict(
        checkpoint["G_OY"]
    )

    D_old.load_state_dict(
        checkpoint["D_old"]
    )

    D_young.load_state_dict(
        checkpoint["D_young"]
    )

    optimizer_G.load_state_dict(
        checkpoint["optimizer_G"]
    )

    optimizer_D_old.load_state_dict(
        checkpoint["optimizer_D_old"]
    )

    optimizer_D_young.load_state_dict(
        checkpoint["optimizer_D_young"]
    )

    return checkpoint["epoch"] + 1