import torch


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_name} was requested, but CUDA is unavailable."
            )
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {device_name} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        device = torch.device("cuda", device_index)
    return device
