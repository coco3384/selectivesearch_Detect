# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import math


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#
#  - Modified version with LBP extractor for texture vectorization


def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """

    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
    
    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

        extract HSV
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image

        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.

        output will be [height(*)][width(*)]
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region

        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # mask by the colour channel
        fd = img[:, colour_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)

    return hist


def _extract_regions(img):

    R = {}

    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
            
            R[l]['bbox_size'] = (R[l]['max_x'] - R[l]['min_x']) * (R[l]['max_y'] - R[l]['min_y'])

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R


    
def _extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    neighbours_mask = numpy.zeros(len(R))

    for cur, a in enumerate(R[:-1]):
        for cur2, b in enumerate(R[cur + 1:]):
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
                neighbours_mask[cur] = True
                neighbours_mask[cur + cur2 + 1] = True

    # test
    for i, there_is_neighbours in enumerate(neighbours_mask):
        if not there_is_neighbours:
            assert R[i] not in [b for (a, b) in neighbours]
            

    return neighbours, neighbours_mask


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        'og_min_x': min(r1["og_min_x"], r2["og_min_x"]),
        'og_min_y': min(r1["og_min_y"], r2["og_min_y"]),
        'og_max_x': max(r1["og_max_x"], r2["og_max_x"]),
        'og_max_y': max(r1["og_max_y"], r2["og_max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    rt['bbox_size'] = (rt['max_x'] - rt['min_x']) * (rt['max_y'] - rt['min_y'])
    return rt

def _expand_regions(regions, border, x_lim, y_lim):
    R = regions.items()
    R_new = {}
    for i, r in enumerate(R):
        r_new = {
            'min_x': max(0, r[1]['min_x'] - border),
            'min_y': max(0, r[1]['min_y'] - border),
            'max_x': min(x_lim, r[1]['max_x'] + border),
            'max_y': min(y_lim, r[1]['max_y'] + border),
            'og_min_x': r[1]['min_x'], 
            'og_min_y': r[1]['min_y'], 
            'og_max_x': r[1]['max_x'], 
            'og_max_y': r[1]['max_y'], 
            'size': r[1]['size'],
            'hist_c': r[1]['hist_c'],
            'hist_t': r[1]['hist_t'],
            'labels': r[1]['labels'],
        }
        r_new['bbox_size'] = (r[1]['max_x'] - r[1]['min_x']) * (r[1]['max_y'] - r[1]['min_y'])
        R_new[i] = r_new
    return R_new


def selective_search(
        im_orig, scale=1.0, sigma=0.8, min_size=50, region_pop=False, max_region_size=3000, border=10):
    '''Selective Search

    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions, region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)
    R = _expand_regions(R, border=border, x_lim=img.shape[1], y_lim=img.shape[0])
    origin_size_of_R = len(R.keys())

    # extract neighbouring information
    neighbours, neighbours_mask = _extract_neighbours(R)

    # R = _crop_no_neigublours_regions(no_neighbours_index, max_region_size)


    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)



    # test

    first_region_to_pop = []
    if region_pop:
        key_to_delete = []
        for t in R.keys():
            if R[t]['bbox_size'] > max_region_size:
                for k, v in list(S.items()):
                    if t in k:
                        if k not in key_to_delete:
                            key_to_delete.append(k)
                first_region_to_pop.append(t)
        for k in key_to_delete:
            del S[k]

    # hierarchal search
    while S != {}:
        # check anny region is larger than prefered regional size
        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        if R[t]['bbox_size'] > max_region_size:
            continue
        else:
            for k in [a for a in key_to_delete if a != (i, j)]:
                n = k[1] if k[0] in (i, j) else k[0]
                S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []

    for k, r in list(R.items()):
        # To Do: 
        if k not in first_region_to_pop:
            regions.append({
                'rect': (
                    r['min_x'], r['min_y'],
                    r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                'bbox_size': r['bbox_size'],
                'size': r['size'],
                'labels': r['labels']
            })

    og_regions = []

    """
    for k, r in list(R.items()):
        if k in R_pop_list:
            og_regions.append({
                'rect': (
                    r['og_min_x'], r['og_min_y'],
                    r['og_max_x'] - r['og_min_x'], r['og_max_y'] - r['og_min_y']),
                'size': r['size'],
                'labels': r['labels']
            })
    """

    return img, regions, og_regions

