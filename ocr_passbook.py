import cv2
import os
import preprocessing
from Levenshtein import jaro
import csv



# path to images folder 
img_paths = os.listdir('data')
# headers of the csv file
fields = ['file', 'name', 'account_no', 'ifsc']
outfile = "data.csv"


# write data as dictionary
with open(outfile, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()

def get_numbers(line):
    '''
    split line there by having list of words
    and then pick the digit element in the list
    '''
    word = ''.join(e for e in line.split() if e.isdigit())
    if word:
        return word
    return 'NaN'


def get_words(words):
    '''
    The Jaro string similarity metric is intended for short strings like personal last names. 
    It is 0 for completely different strings and 1 for identical strings.
    eg: jaro('Brian', 'Jesus') ==> 0.0
    eg: jaro('Thorkel', 'Thorgier') ==> 0.779761...

    read more here
    https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html


    the purpose is to find similar words if ocr misses or mismatches few character

    '''
    d = {}
    for word in words.split():
        if jaro('name', word) > 0.9:
            d['name'] = word
        if jaro('account', word) > 0.9:
            d['account'] = word
        if jaro('ifsc', word) > 0.9:
            d['ifsc'] = word
    return d


def get_name(line, key):
    '''
    get name of the account holder
    '''
    if ':' in line:
        word = line.split(':')[-1].strip()
        return word
    elif '=' in line:
        word = line.split('=')[-1].strip()
        return word 
    else:
        line = line.split(key)[-1]
        if len(line.strip().split()) <= 5:
            return line
    return 'NaN'


def get_ifsc(line, key):
    '''
    return ifsc code form the line 
    ifsc code is always of 11 characters hence the check for length of ifsc
    '''
    if ':' in line:
        word = line.split(':')[-1].strip()
        if len(word) == 11:
            return word
    elif '=' in line:
        word = line.split('=')[-1].strip()
        if len(word) == 11:
            return word
    else:
        line = line.split(key)[-1]
        for word in line.split():
            if len(word) == 11:
                return word
    return 'NaN'
    


def get_ocr(img, data_dict):
    text = str(preprocessing.detect(img))
    # The recognized text is stored in variable text
    # Any string processing may be applied on text
    # Here, basic formatting has been done:

    # iterate over the text from the tesseract line by line
    for word in text.split('\n'):
        
        word = word.lower()

        # check if words are similar by using Levenshtein algo
        values = get_words(word)
        for key in values:
            if key == 'account':
                # get the digits for the account number
                number = get_numbers(word)
                # process only if value is NaN
                if data_dict['account_no'] == 'NaN' or len(data_dict['account_no']) < 8:
                    data_dict['account_no'] = number if  number != 'NaN' else 'NaN'
            if key == 'name':
                # process only if value is NaN
                if data_dict['name'] == 'NaN':
                    # get name
                    data_dict['name'] = get_name(word, values[key])
                    update_name = ''
                    # if name contains any special character remove them
                    if data_dict['name'] != 'NaN':
                        for name in data_dict['name'].split():
                            if name.isalpha():
                                update_name += name+' '
                        data_dict['name'] = update_name if update_name else 'NaN'

                    
            if key == 'ifsc':
                # process only if value is NaN
                if data_dict['ifsc'] == 'NaN':
                    data_dict['ifsc'] = get_ifsc(word, values[key])
                    

for img_path in img_paths:
    # dict to store desired values & to write it to csv
    data_dict = {}
    data_dict['file'] = img_path
    data_dict['name'] = 'NaN'
    data_dict['account_no'] = 'NaN'
    data_dict['ifsc'] = 'NaN'


    img_path = os.path.join('data', img_path)
    img = cv2.imread(img_path)
    # preprocessing.show_image(img)
    print(f'{img_path} {img.shape}')

    # resize only if image is too small
    if img.shape[0] < 1000:
        img = preprocessing.resize(img)

    # brighten the dark images by increase alpha & beta value automatically 
    auto_bright, _, _ = preprocessing.automatic_brightness_and_contrast(img)

    # get the current orientation of the image and 
    # angle required to rotate if image is not properly oriented
    angle = preprocessing.get_angle(img)
    angle_counter = 90
    while angle != 0:
        img = preprocessing.rotate(img, angle)

        # sometimes even after rotating with the angle images are
        # not in the desired orientation hence we rotate till the angle value is 0
        angle = preprocessing.get_angle(img)
        if angle != 0:
            angle = angle_counter
            angle_counter += 90

    # preprocess the images and get ocr at different preprocessing stages
    # different processing works better on different types of images
    # hence increasing preprocessing complexity linearly
    img = preprocessing.increase_brightness(img)
    img_ = preprocessing.contrast_brightness(img)
    get_ocr(img_, data_dict)
    img_ = preprocessing.get_grayscale(img_)
    get_ocr(img_, data_dict)
    img_ = preprocessing.get_binary(img_)
    get_ocr(img_, data_dict)
    img = preprocessing.get_grayscale(img)
    get_ocr(img, data_dict)
    img = preprocessing.unsharp_mask(img)
    get_ocr(img, data_dict)
    img = preprocessing.increase_brightness(auto_bright)
    img = preprocessing.get_grayscale(img)
    get_ocr(img, data_dict)
    # print(data_dict)

    # write to the data collected to the csv file
    with open(outfile, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writerow(data_dict)