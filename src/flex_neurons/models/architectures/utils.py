"""
This is a way to trace the in and out dimensions of Torch conv modules.
"""


class DimensionTracer:
    def __init__(self, initial_image_dimensions):
        self.initial_image_dimensions = initial_image_dimensions
        self.registry = []

    def __call__(self, **kwargs):
        self.registry.append(kwargs)

    def calculate_dimension(self):
        c, w, h = self.initial_image_dimensions

        for operation in self.registry:
            if "in_channels" in operation and "out_channels" in operation:
                F = operation.get("kernel_size", 1)
                P = operation.get("padding", 0)
                S = operation.get("stride", 1)
                c = operation["out_channels"]
                w = (w - F + 2 * P) // S + 1
                h = (h - F + 2 * P) // S + 1
            elif "kernel_size" in operation:
                F = operation.get("kernel_size", 1)
                S = operation.get("stride", 1)
                w = (w - F) // S + 1
                h = (h - F) // S + 1

        return c, w, h
