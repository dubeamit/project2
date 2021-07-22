import pytesseract
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from PIL import Image


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def replace_chars(text):
    """
    Replaces all characters instead of numbers from 'text'.
    
    :param text: Text string to be filtered
    :return: Resulting number
    """
    list_of_numbers = re.findall(r'\d+', text)
    result_number = ''.join(list_of_numbers)
    return result_number


def resize(img, width=960):
    '''
    resize while maintaining aspect ratio
    '''
    return imutils.resize(img, width=width)

def get_binary(image):
    '''
    get a binarized image using thresholding 
    '''
    _, img_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # if np.mean(img_bin) > 127:
    #     img_bin = cv2.bitwise_not(img_bin)
    img_bin = cv2.bitwise_not(img_bin)
    return img_bin

def bin_wolf_julion(image):
    '''
    binarized image:  Wolf-Julion local binarization
    '''
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2]

    T = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=81, k=0.1, binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
    img = (img > T).astype("uint8") * 255
    return img
    
def detect(cropped_frame, is_number = False):
    '''
    tesseract ocr
    '''
    if (is_number):
        text = pytesseract.image_to_string(cropped_frame, config='digits')
                                        #    config ='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 2')
    else:
        text = pytesseract.image_to_string(cropped_frame)        
        
    return text


#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def erode(image):
    '''
    erode foregroud boundaries & diminish the features of an image
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_erode = cv2.erode(image, kernel, iterations = 2)
    return img_erode

def dilate(image):
    '''
    increase the object area & accentuate features
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_dilate = cv2.dilate(image, kernel, iterations = 2)
    return img_dilate


def blur(img, blur_weight=1):
    '''
    smoothing the images
    '''
    img_blurred = cv2.medianBlur(img, blur_weight)
    return img_blurred


def sharpen(image):
    '''
    sharpen the image: highlights edges and fine details in the image
    '''
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def contrast_brightness(img, contrast = 1.5, brightness = 50):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:,:,2] = np.clip(contrast * img[:,:,2] + brightness, 0, 255)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def rotate(image, angle):
    return imutils.rotate_bound(image, angle)


def get_angle(img):
    return int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(img)).group(0))


def change_dpi(img, path):
    '''
    increasing dpi of the image
    '''
    RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    PILimage = Image.fromarray(RGBimage)
    path = path.rsplit('.', 1)[0] + '_dpi.jpg'
    PILimage.save(f'{path}', dpi=(500,500))
    return path


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    '''
    find alpha(contrast control) and beta(brightness control) values automatically

    To calculate alpha, we take the minimum and maximum grayscale range after clipping 
    and divide it from our desired output range of 255

    To calculate beta, we plug it into the formula where g(i, j)=0 and f(i, j)=minimum_gray
    g(i,j) = α * f(i,j) + β
    after sloving β = -minimum_gray * α
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


def show_image(image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.show()



if __name__ == '__main__':
    img = cv2.imread('data/3_dpi.jpg')
    # img, alpha, beta = automatic_brightness_and_contrast(img)
    # print('alpha', alpha)
    # print('beta', beta)
    # cv2.imshow('auto_result', img)
    img = imutils.resize(img, width=960)
    img = increase_brightness(img)
    img = unsharp_mask(img)
    img = get_grayscale(img)
    
    print('og\n', detect(img))
    # cv2.imwrite('auto_result.png', auto_result)
    cv2.imshow('image', img)
    cv2.waitKey()
    exit()
    # img = cv2.imread('data/ind`ian-bank-cif.jpg')
    img = cv2.imread('data/4.jpg')
    # img = cv2.imread('data/3.jpeg')
    img = imutils.resize(img, width=960)
    cv2.imshow('img', img)

    angle = int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(img)).group(0))
    if angle != 0:
        print('angle', angle)
        img = rotate(img, angle)

    # exit()
    img = increase_brightness(img)
    img = get_grayscale(img)
    print('og\n', detect(img))
    # show_image(img)
    cv2.imshow('og_img', img)

    img_u = unsharp_mask(img)
    print('un_sharpen\n', detect(img_u))
    cv2.imshow('un_sh_img', img_u)

    img_c = unsharp_mask(img)
    print('contrast\n', detect(img_c))
    cv2.imshow('contrast', img_c)

    cv2.waitKey(0)
    # show_image(img)
    # exit()
    