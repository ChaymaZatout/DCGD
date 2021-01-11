import numpy as np
from Components.Camera import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys
import cv2


class DCGD:

    def __init__(self, camera: Camera, interval, height_err, size_err, step=1):
        self._height_err = height_err
        self._interval = interval
        self._camera = camera
        self._size_err = size_err
        self._step = step

    # def compute_cuts(self, depthImg, interval, cy_d=2.4273913761751615e+02, fy_d=5.9104053696870778e+02):
    #     print("compute_cuts: creating cuts matrix ...")
    #     (height, width) = depthImg.shape[:2]
    #     cuts = np.full((len(interval), width), np.nan)
    #     print("compute_cuts: cuts computing")
    #     for x in range(width):
    #         for y in range(height):
    #             curr = depthImg[y, x]  # get tue current depth
    #             if curr in interval:  # if the current depth is valable
    #                 # compute real_y
    #                 y_real = -((y - cy_d) * curr / fy_d)  # compute the real depth
    #                 ind = curr - min(interval)  # compute the cut indice
    #                 if math.isnan(cuts[ind, x]):
    #                     cuts[ind, x] = y_real
    #                 elif y_real < cuts[ind, x]:
    #                     cuts[ind, x] = y_real
    #     return cuts

    def compute_subcuts(self, cut):
        sub_cuts = []
        if len(cut) < 2:
            return []
        sub = [cut[0]]
        for i in range(1, len(cut)):
            diff = abs(cut[i - 1][2] - cut[i][2])
            if diff < self._height_err:
                sub.append(cut[i])
            else:
                sub_cuts.append(sub)
                sub = [cut[i]]
        if len(sub) != 0:
            sub_cuts.append(sub)
        return sub_cuts


    def compute_all_subcuts(self, cuts):
        subcuts = []
        for cut in cuts:
            subcuts.append(self.compute_subcuts(cut))
        return subcuts


    def compute_subcuts_q(self, cut, q=50):
        sub_cuts = []
        if len(cut) < 2:
            return []
        sub = [cut[0]]
        for i in range(1, len(cut)):
            # list of y real:
            yreals = [el[2] for el in sub]
            # get quantile:
            quantile = np.percentile(yreals, q)
            diff = abs(quantile - cut[i][2])
            if diff < self._height_err:
                sub.append(cut[i])
            else:
                sub_cuts.append(sub)
                sub = [cut[i]]
        if len(sub) != 0:
            sub_cuts.append(sub)
        return sub_cuts

    def filter_subcuts(self, subcuts):
        if len(subcuts) == 0:
            return [], []

        filtered = []
        noise = []
        i = 0
        while i < len(subcuts) and len(subcuts[i]) < self._size_err:
            noise.append(subcuts[i])
            i += 1
        if i == len(subcuts):
            return [], noise

        filtered.append(subcuts[i])
        i += 1
        while i < len(subcuts):
            subcut = subcuts[i]
            if len(subcut) >= self._size_err:
                filtered.append(subcut)
            else:
                noise.append(subcut)
                if i + 1 < len(subcuts):
                    if abs(filtered[-1][-1][2] - subcuts[i + 1][0][2]) < self._height_err:
                        filtered[-1] += subcuts[i + 1]
                        i += 1
            i += 1

        if i != len(subcuts):
            subcut = subcuts[i]
            if len(subcut) >= self._size_err:
                filtered.append(subcut)
            else:
                noise.append(subcut)

        return filtered, noise

    def check_and_update_annotation(self, annotations, i):
        if annotations[i][0] == -1:
            sub = annotations[i][1]
            subp = annotations[i + 1][1]
            if sub[-1][2] < subp[0][2]:
                annotations[i][0] = "cc"
                annotations[i + 1][0] = "cv"
            else:
                annotations[i][0] = "cv"
                annotations[i + 1][0] = "cc"

        elif annotations[i][0] == "cc":
            sub = annotations[i][1]
            subp = annotations[i + 1][1]
            if sub[-1][2] < subp[0][2]:
                annotations[i + 1][0] = "cv"
            else:
                annotations[i][0] = "cv"
                annotations[i + 1][0] = "cc"

        elif annotations[i][0] == "cv":
            sub = annotations[i][1]
            subp = annotations[i + 1][1]
            if sub[-1][2] < subp[0][2]:
                annotations[i + 1][0] = "cv"
            else:
                annotations[i + 1][0] = "cc"

        return annotations[i], annotations[i + 1]

    def label_subcuts(self, subcuts):
        if len(subcuts) == 1:
            return [["cc", subcuts[0]]]
        if len(subcuts) == 0:
            return []

        # initialization:
        annotations = [[-1, sc] for sc in subcuts]
        for i in range(len(annotations) - 1):
            annotations[i], annotations[i + 1] = self.check_and_update_annotation(annotations, i)
            
        return annotations

    def label_subcuts_ymin(self, subcuts, miny, error_int):
        if len(subcuts) == 1:
            return [["cc", subcuts[0]]]
        if len(subcuts) == 0:
            return []

        # initialisation:
        annotations = [[-1, sc] for sc in subcuts]

        for i in range(len(annotations) - 1):
            mean_y = np.mean([cut[2] for cut in annotations[i][1]])  # compute the y mean of the current subcut;
            if mean_y > miny + error_int:  # if the y exceeds the floor (miny of all the scene)
                annotations[i][0] = "cv"
            else:
                annotations[i], annotations[i + 1] = self.check_and_update_annotation(annotations, i)
            
        return annotations

    def label_all_subcuts(self, allsubcuts):
        allannotations = []
        for subcuts in allsubcuts:
            allannotations.append(self.label_subcuts(subcuts))
        return allannotations

    def filter_all_subcuts(self, allsubcuts):
        allfiltered = []
        allnoise = []
        for subcuts in allsubcuts:
            filtered, noise = self.filter_subcuts(subcuts)
            allfiltered.append(filtered)
            allnoise.append(noise)
        return allfiltered, allnoise

    def compute_cuts_downsampling_miny(self, depthImg):

        (height, width) = depthImg.shape[:2]
        (mini, maxi) = self._interval
        cuts = [[]] * int((maxi - mini + 1) / self._step + 1)
        boundary = len(cuts)

        arr_depthInd = ((depthImg - mini) / self._step).astype(int)

        filtre1 = arr_depthInd >= 0
        filtre2 = arr_depthInd < boundary
        arr_depthInd = np.where(filtre1 & filtre2, arr_depthInd, np.nan)

        Y_indices = np.arange(height)
        Y_indices = - (Y_indices - self._camera.cy_d)
        arr_yReal = depthImg * Y_indices[:, None] / self._camera.fy_d
        
        # append the smallest height for each x to the right cut indice in the cuts list
        for i in range(boundary):
            arr = np.where(arr_depthInd == i, arr_yReal, np.nan)
            min_Yreals_Ind = pd.DataFrame(arr).idxmin()
            min_Yreals_Ind = min_Yreals_Ind.dropna()
            
            if min_Yreals_Ind.size > 0:
                X = list(min_Yreals_Ind.keys())
                Y = list(min_Yreals_Ind.values.astype(int))
                Yreals = list(arr[Y, X])
                cuts[i] = list(zip(X, Y, Yreals))
        # compute miny: the smallest height in the entire scene
        miny = np.min(arr_yReal)
        
        return cuts, miny

    def cgd_process_downsampling(self, depthImg):
        
        (mini, maxi) = self._interval
        lenList = int((maxi - mini + 1) / self._step + 1)

        print("Computing Cuts...")
        start = time.time()
        cuts, miny = self.compute_cuts_downsampling(depthImg)
        end = time.time()
        print(f'Consumed time: {end - start}')
        print("Cuts Computed.\n")
        print("Computing All Subcuts...")
        subcuts = self.compute_all_subcuts(cuts)
        print('All Subcuts Computed.\n')
        print('Filtering All Subcuts...')
        filtered, noise = self.filter_all_subcuts(subcuts)
        print('All Subcuts Filtered.\n')
        print('Labeling All Filtered Subcuts...')
        labels = self.label_all_subcuts(filtered)
        print('All Filtered subcuts Labeled.\n')
        print('Labeling All Subcuts...')
        labelsn = self.label_all_subcuts(subcuts)
        print('All subcuts Labeled.')

        # Get minimal floor points, the ones annotated with 'cc' per depth indice
        floorPoints = [[point for a in labels[ind] if a[0] == 'cc' for point in a[1]] for ind in range(lenList)]

        return labels, labelsn, noise, floorPoints
    def compute_transformationMatrix(self, allsubcuts, minSize=10):
        mmse = sys.float_info.max
        mx = []
        my = []
        myr = []
        msize = 0

        reg = LinearRegression()
        for subcuts in allsubcuts:
            for sc in subcuts:
                if len(sc)> 3:
                    x = [p[0] for p in sc]
                    y = [p[2] for p in sc]

                    # fitting:
                    reg.fit(np.column_stack([x,y]), y)

                    # compute mse:
                    yr = reg.predict(np.column_stack([x,y]))
                    mse = mean_squared_error(y, yr)

                    if mse < mmse and len(sc)> minSize:
                        mmse = mse
                        mx = x
                        my = y
                        myr = yr

        # compute src :
        i = int((len(mx) + 1) / 2)
        src = [[mx[i - 1], my[i - 1]], [mx[i], my[i]], [mx[i + 1], my[i + 1]]]
        dst = [[mx[i - 1], myr[i - 1]], [mx[i], myr[i]], [mx[i + 1], myr[i + 1]]]
        # compute affine:
        return cv2.getAffineTransform(np.float32(src), np.float32(dst))

    def map_subcuts(self, subcuts, affinM):
        affsubcuts = []
        for sc in subcuts:
            src = [[p[0], p[2], 1] for p in sc]
            res = np.array(affinM).dot(np.array(src).transpose())
            resd = [[p[0], p[1]] for p in sc]
            resg=[r[1] for r in list(np.array(res).transpose())]
            # print(list(np.column_stack([resd, resg])))
            affsubcuts.append(np.column_stack([resd, resg]).tolist())

        return affsubcuts


    def map_allsubcuts(self, allsubcuts, affinM):
        allaffines = []
        for scs in allsubcuts:
            allaffines.append(self.map_subcuts(scs, affinM))
        return allaffines
