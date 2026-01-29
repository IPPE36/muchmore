import numpy as np
from source.field import dump_field, load_field, label_field
from source.mesh_2D import mesh_2d
from source.mesh_3D import mesh_3d


class Microstructure:
    def __init__(self, field: np.ndarray, size: float = 30):
        self.field = field
        self.shape = field.shape
        self.ndim = field.ndim
        self.size = size
        self.vf = np.count_nonzero(field == 1) / field.size

    def mesh(self):
        if self.ndim == 2:
            mesh_2d(self.field)
        elif self.ndim == 3:
            mesh_3d(self.field)
        else:
            raise NotImplementedError

    def dump(self, filepath: str):
        dump_field(self.field, filepath)

    @classmethod
    def load(cls, filepath: str, size: float = 30) -> "Microstructure":
        field = load_field(filepath)
        return cls(field=field, size=size)


if __name__ == "__main__":
    exit()