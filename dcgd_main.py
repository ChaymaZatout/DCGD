"""
Author : Chayma Zatout
Execution time optimized by : Racha Salhi
"""
from Components.DCGD import *
from Components.Visualizer import *
import cv2


# depth image rendering to draw colors:
def depth_to_rgb(depth):
    (height, width) = (480, 640)
    rgb = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            rgb[y, x] = (depth[y, x], depth[y, x], depth[y, x])

    return rgb


# create image intensity;
def pretty_depth(d):
    depth = np.copy(d)
    np.clip(depth, 0, 2 ** 10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth_to_rgb(depth)


if __name__ == '__main__':
    results_dir = "_out/"

    depth = cv2.imread('_in/0_d.png', cv2.CV_16U)
    pretty = cv2.imread('_in/0_p.png')

    # Camera parameters:
    cy_depth = 2.3844389626620386e+02
    fy_depth = 5.8269103270988637e+02
    cam = Camera(cy_depth, fy_depth)
    interval = (800, 4000)

    # create a DCGD object
    h_err = 20
    size_err = 3
    step = 15
    dcgd = DCGD(cam, interval, h_err, size_err, step)

    # floor detection:
    labels, labelsn, noise, minimalFloorPoints = dcgd.cgd_process_downsampling(depth)

    # Floor visualization
    start = time.time()
    floorPoints = Visualizer.viz_on_depth_downsampling(pretty, depth, minimalFloorPoints, interval,
                                                       h_err, step, cy_depth, fy_depth)
    end = time.time()
    print(f'Consumed time: {end - start}')

    # save floor:
    pretty = cv2.medianBlur(pretty, 11)
    cv2.imwrite(results_dir + "floor.png", pretty)
