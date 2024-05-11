import base64
import math
from io import BytesIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import feret
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import pandas as pd

#Set default matplotlib colormap to gray
mpl.rc('image', cmap='gray')

def cropToIndividualOrganoids(image, labels, multiplyFeret=2, deleteEdgeTouching=True, outSize=512, mode="PAD", manualScaleFactor=-1):
    '''
    Parameters:
        image (array): The raw image that is being labeled
        labels (array): Labels for individual organoids with each organoid labeled a different integer {1,2,...}
        multiplyFeret (int): What to multiply the maximum feret (caliper) distance by for the crop sized
        deleteEdgeTouching (bool): If true, deletes the organoids touching the edge of the image
        outSize (int): The size in pixels of the output images (128, 256, 512, 1024)
        mode (str): The method of padding images (PAD, RESIZE_WITH_PADDING)
        manualScaleFactor (int): If this number is greater than 0, every image will be scaled manually (good for datasets where you want to take size of the organoid into account)

    Returns:
        individualOrganoidList (list): A list of cropped image arrays containing an individual organoid. Each element of the list is a two-element list, with the first element being the cropped base image and the second being the organoid mask
    '''
    individualOrganoidList = []
    individualOrganoidListScaled = []
    scaleFactor = 1

    for i in list(set(labels.flatten())):
        if i == 0: continue

        # Mask the whole image to individual organoids
        mask = np.where(labels==i, 1, 0)
        # print(set(mask.flatten()))
        # print(np.shape(image), np.shape(mask))
        maskedIm = np.multiply(image, mask)

        #Check if the mask is touching the edge of the image
        if deleteEdgeTouching:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            if x<=0 or y<=0 or x+w>=np.shape(mask)[1] or y+h>=np.shape(mask)[0]: continue

        #Crop the whole image to smaller images centered at the medioid of the masks
        x, y = calcCentroid(mask)
        r = int(calcMaxFeret(mask)*multiplyFeret)
        xmin = int(x)-r//2 if int(x)-r//2>=0 else 0
        xmax = int(x)+r//2 if int(x)+r//2<=np.shape(mask)[0] else np.shape(mask)[0]
        ymin = int(y) - r // 2 if int(y) - r // 2 >= 0 else 0
        ymax = int(y) + r // 2 if int(y) + r // 2 <= np.shape(mask)[1] else np.shape(mask)[1]
        cropMaskedIm = maskedIm[xmin:xmax, ymin:ymax]
        cropMask = mask[xmin:xmax, ymin:ymax]
        # plt.figure()
        # plt.imshow(cropMaskedIm)

        #Check if cropMaskedIm is empty
        if not np.any(cropMaskedIm): continue

        #Update the max size of cropped images
        if max(np.shape(cropMaskedIm)) > outSize/scaleFactor:
            scaleFactor = outSize/max(np.shape(cropMaskedIm))

        #Add unscaled image to organoid list
        # plt.figure()
        # plt.imshow(cropMaskedIm)
        individualOrganoidList.append([cropMaskedIm.copy(), cropMask.copy()])

    recScaleFactor = scaleFactor
    if manualScaleFactor>0:
        scaleFactor = manualScaleFactor

    # Rescale all the images by scaleFactor and pad
    for org in individualOrganoidList:
        resizeIm = resizeImage(org[0], scaleFactor)
        resizeMask = resizeImage(org[1], scaleFactor)

        if mode == "RESIZE_WITH_PADDING":
            resizeIm = resizeAndPadImage(resizeIm.astype('float32'), (outSize, outSize))
            resizeMask = resizeAndPadImage(resizeMask.astype('float32'), (outSize, outSize))
        elif mode == "PAD":
            resizeIm = padImage(resizeIm, (outSize, outSize))
            resizeMask = padImage(resizeMask, (outSize, outSize))
        else:
            print("Mode not accepted. Acceptable modes are: RESIZE_WITH_PADDING, PAD. Using default of PAD.")
            resizeIm = padImage(resizeIm, (outSize, outSize))
            resizeMask = padImage(resizeMask, (outSize, outSize))

        #plt.figure()
        #plt.imshow(resizeIm)
        # plt.imshow(resizeMask)
        individualOrganoidListScaled.append([resizeIm, resizeMask])

    # showImageList([x[0] for x in individualOrganoidListScaled], 8)
    return individualOrganoidListScaled, recScaleFactor

def showImageList(imageList, numColumns):
    fig = plt.figure(figsize=(40., 40.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(imageList) // numColumns + 1, numColumns),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes
                     )

    for ax, im in zip(grid, imageList):
        ax.imshow(im)
        ax.set_axis_off()

    return fig

def resizeAndPadImage(image, new_shape, padding_color=(0, 0, 0)):
    """Maintains aspect ratio and resizes with padding.
    Parameters:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    print(ratio)
    new_size = tuple([int(x*ratio) for x in original_shape])
    print(new_size)
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def padImage(image, new_shape, padding_color=(0,0,0)):
    if np.shape(image) == new_shape: return image
    delta_w = new_shape[1] - np.shape(image)[1]
    delta_h = new_shape[0] - np.shape(image)[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def resizeImage(image, scale):
    image = image.astype("float32")
    shp = np.shape(image)
    outIm = cv2.resize(image, (round(shp[1]*scale), round(shp[0]*scale)))
    return outIm

def calcCentroid(binaryMask):
    count = (binaryMask == 1).sum()
    x_center, y_center = np.argwhere(binaryMask == 1).sum(0) / count
    return (x_center, y_center)

def calcMinFeret(binaryMask):
    retVal = -1
    try:
        retVal = feret.min(binaryMask)
    except (IndexError, ValueError) as err:
        pass
    return retVal

def calcMaxFeret(binaryMask):
    retVal = -1
    try:
        retVal = feret.max(binaryMask)
    except (IndexError, ValueError) as err:
        pass
    return retVal

def calcSizeXY(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (w,h)

def calcAreaAndPerimeter(contour):
    return (cv2.contourArea(contour), cv2.arcLength(contour, True))

def calcConvexAreaAndPerimeter(contour):
    convexHull = cv2.convexHull(contour)
    return (cv2.contourArea(convexHull), cv2.arcLength(convexHull, True))

def calcFormFactorAndRoundness(contour):
    area, perimeter = calcAreaAndPerimeter(contour)
    convexArea, convexPerimeter = calcConvexAreaAndPerimeter(contour)
    formFactor = 4*math.pi*area/(perimeter**2) if perimeter != 0 else -1
    roundness = 4*math.pi*area/(convexPerimeter**2) if convexPerimeter != 0 else -1
    return (formFactor, roundness)

def calcSolidity(contour):
    area, _ = calcAreaAndPerimeter(contour)
    convexArea, _ = calcConvexAreaAndPerimeter(contour)
    if convexArea != 0:
        return area/convexArea
    else:
        return-1

def calcConvexityDefects(contour):
    #Roughly the number of buds
    convexHull = cv2.convexHull(contour, returnPoints=False)
    convexityDefects = cv2.convexityDefects(contour, convexHull)
    return len(convexityDefects) if convexityDefects is not None else 0

def calcEllipse (contour):
    if len(contour) >= 5:
        (x, y), (ma, MA), angle = cv2.fitEllipse(contour)
        eccentricity = ma/MA
        return (x,y,MA,ma,angle,eccentricity)
    else:
        return (-1,-1,-1,-1,-1,-1)

def calcConvexity(contour):
    _, perimeter = calcAreaAndPerimeter(contour)
    _, convexPerimeter = calcConvexAreaAndPerimeter(contour)
    if perimeter != 0:
        return convexPerimeter/perimeter
    else:
        return -1

def calcIntensity(image):
    return np.sum(image)

def calcMeanIntensity(contour, image):
    area, _ = calcAreaAndPerimeter(contour)
    if area != 0:
        return calcIntensity(image)/area
    else:
        return -1

def toPng(filepath, mult=1):
    out = BytesIO()
    with Image.open(filepath) as im:
        imarr = np.asarray(im)
        imarrnorm = imarr*mult
        im = Image.fromarray(imarrnorm)
        im = im.convert("L")
        im.save(out, format='png')
    return out.getvalue()

def b64ImageFiles(images, colormap='magma'):
    urls = []
    for im in tqdm(images):
        #png = to_png(img_as_ubyte(cmap(im)))
        png = toPng(im)
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
        urls.append(url)
    return urls