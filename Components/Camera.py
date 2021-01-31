"""
Author : Chayma Zatout
"""
class Camera:

    def __init__(self, cy_depth, fy_depth, cx_depth=0, fx_depth=0):
        self._cy_d = cy_depth
        self._fy_d = fy_depth
        self._cx_d = cx_depth
        self._fx_d = fx_depth

    @property
    def cy_d(self):
        return self._cy_d

    @property
    def fy_d(self):
        return self._fy_d

    @property
    def fx_d(self):
        return self._fx_d

    @property
    def cx_d(self):
        return self._cx_d

