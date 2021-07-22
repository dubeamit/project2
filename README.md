# project2
## OCR on passbook

### Data directory
1. make a folder **data** and put all the images in it

### Generating data (ocr_passbook.py)
#### run `python3 ocr_passbook.py` to generated data 
1. preprocessing the images 
2. do OCR on images and generate text
3. write name, account number, ifsc to a csv file 



### Preprocessing 
1. **Resizing**: resize the images while maintaining the aspect ratio
2. **increase brightness**: change the only the brightness of image in **hsv** space
3. **Blur**: smoothing the images
4. **automatically increase brightness and contrast**: find alpha(contrast control) and beta(brightness control). <br>
    Brightness and contrast can be adjusted using alpha (α) and beta (β), respectively. The expression can be written as `g(i,j) = α * f(i,j) + β` <br>
    values automatically to calculate alpha, we take the minimum and maximum grayscale range after clipping and divide it from our desired output range of 255 <br>
    To calculate beta, we plug it into the formula where <br>
    `g(i, j)=0 and f(i, j)=minimum_gray` <br>
    `g(i,j) = α * f(i,j) + β` <br>
    after sloving <br>
    `β = -minimum_gray * α`
5. **unsharp_mask**: The unsharp filter is a simple sharpening operator which derives its name from the fact that it enhances edges (and other high frequency components in an image) via a procedure which subtracts an unsharp, or smoothed, version of an image from the original image. The unsharp filtering technique is commonly used in the photographic and printing industries for crispening edges. [read more here](https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm)


### Requirements
1. all requirements are listed in requirements.txt
2. few apt packages are include in requirements install those and remove from the requirements.txt
3. run `pip3 install -r requirements.txt`