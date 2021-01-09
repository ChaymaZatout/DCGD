class Camera:

    def __init__(self, cy_depth, fy_depth):
        self._cy_d = cy_depth
        self._fy_d = fy_depth

    @property
    def cy_d(self):
        return self._cy_d

    @property
    def fy_d(self):
        return self._fy_d


