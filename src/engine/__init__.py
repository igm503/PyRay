def get_engine(device: str):
    if device == "cpu":
        from .cpu import CPUTracer

        return CPUTracer()
    elif device == "metal":
        from .metal import MetalTracer

        return MetalTracer()
    elif device == "cuda":
        from .cuda import CudaTracer

        return CudaTracer()
    else:
        raise ValueError(f"Unknown device {device}")
