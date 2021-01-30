"""
Author : Chayma Zatout
Execution time optimized by : Racha Salhi
"""
import cv2
import freenect
from Components.DCGD import *
from Components.Visualizer import *


# function to get RGB image from kinect:
def get_rgb():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    return array


# function to get depth image from kinect:
def get_depth():
    array = freenect.sync_get_depth()[0]
    array = array.astype(np.uint16)
    return array


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

    results_dir = ""
    # displaying videos:
    cv2.namedWindow('Depth')
    cv2.namedWindow('RGB')

    while 1:
        # get a frame from RGB camera
        rgbimg = get_rgb()
        # display RGB image
        cv2.imshow('RGB', rgbimg)

        # get a frame from depth sensor
        depth = get_depth()

        # display depth image
        cv2.imshow('Depth', depth)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    # save image:
    pretty = pretty_depth(depth)
    cv2.imwrite(results_dir + "pretty.png", pretty)
    cv2.imwrite(results_dir + "rgb.png", rgbimg)

    # Camera parameters:
    cy_depth = 2.3844389626620386e+02
    fy_depth = 5.8269103270988637e+02
    cam = Camera(cy_depth, fy_depth)
    interval = (800, 4000)

    # ground detection:

    # create a DCGD object
    h_err = 20
    size_err = 5
    step = 15
    dcgd = DCGD(cam, interval, h_err, size_err, step)

    # floor detection: ########################################################################
    cuts, ymin = dcgd.compute_cuts_downsampling_miny(depth)
    subcuts = dcgd.compute_all_subcuts(cuts)
    affM = dcgd.compute_transformationMatrix(subcuts) # changement de repere
    subcuts = dcgd.map_allsubcuts(subcuts, affM)

    ############################################################################################
    filtered, noise = dcgd.filter_all_subcuts(subcuts)
    # labels = dcgd.label_all_subcuts(filtered)
    # labelsn = dcgd.label_all_subcuts(subcuts)

    labels = dcgd.label_all_subcuts_miny(filtered, ymin, 10)
    labelsn = dcgd.label_all_subcuts_miny(subcuts, ymin, 10)
    
    # Get minimal floor points of the labels set, the ones annotated with 'cc' per depth indice
    floorPoints_labels = [[point for a in labels[ind] if a[0] == 'cc' for point in a[1]] for ind in range(lenList)]
    
    # Get minimal floor points of the labelsn set, the ones annotated with 'cc' per depth indice
    floorPoints_labelsn = [[point for a in labelsn[ind] if a[0] == 'cc' for point in a[1]] for ind in range(lenList)]

    p = np.copy(pretty)
    # visualization:
    Visualizer.viz_all_labels(labelsn, interval, show=True, filename=results_dir +"labels.png")
    Visualizer.viz_all_filtered_labels(labels, noise, interval, show=True, filename=results_dir +"filtered.png")
    Visualizer.viz_on_depth_downsampling(pretty, depth, floorPoints_labels, interval, h_err, step, noise, cy_depth, fy_depth)
    Visualizer.viz_on_depth_downsampling(p, depth, floorPoints_labelsn, interval, h_err, step, noise, cy_depth, fy_depth)


    # save floor:
    cv2.imwrite(results_dir + "floor_f.png", pretty)
    cv2.imwrite(results_dir + "floor.png", pretty)
    cv2.imshow("Floor", pretty)
    cv2.waitKey(0)
