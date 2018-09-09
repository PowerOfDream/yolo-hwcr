import glob
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import gen_bg_img

def load_image_file_names(char_list):
    '''
    load image file names from image_path

    char_list: character list

    return: a diction mapping from character to image file names
    '''

    image_path = {
        '0': "C:/YOLO-OCR/dataset/30/train_30",
        '1': "C:/YOLO-OCR/dataset/31/train_31",
        '2': "C:/YOLO-OCR/dataset/32/train_32",
        '3': "C:/YOLO-OCR/dataset/33/train_33",
        '4': "C:/YOLO-OCR/dataset/34/train_34",
        '5': "C:/YOLO-OCR/dataset/35/train_35",
        '6': "C:/YOLO-OCR/dataset/36/train_36",
        '7': "C:/YOLO-OCR/dataset/37/train_37",
        '8': "C:/YOLO-OCR/dataset/38/train_38",
        '9': "C:/YOLO-OCR/dataset/39/train_39",
        'A': "C:/YOLO-OCR/dataset/41/train_41",
        'B': "C:/YOLO-OCR/dataset/42/train_42",
        'C': "C:/YOLO-OCR/dataset/43/train_43",
        'D': "C:/YOLO-OCR/dataset/44/train_44",
        'E': "C:/YOLO-OCR/dataset/45/train_45",
        'F': "C:/YOLO-OCR/dataset/46/train_46",
        'G': "C:/YOLO-OCR/dataset/47/train_47",
        'H': "C:/YOLO-OCR/dataset/48/train_48",
        'I': "C:/YOLO-OCR/dataset/49/train_49",
        'J': "C:/YOLO-OCR/dataset/4a/train_4a",
        'K': "C:/YOLO-OCR/dataset/4b/train_4b",
        'L': "C:/YOLO-OCR/dataset/4c/train_4c",
        'M': "C:/YOLO-OCR/dataset/4d/train_4d",
        'N': "C:/YOLO-OCR/dataset/4e/train_4e",
        'O': "C:/YOLO-OCR/dataset/4f/train_4f",
        'P': "C:/YOLO-OCR/dataset/50/train_50",
        'Q': "C:/YOLO-OCR/dataset/51/train_51",
        'R': "C:/YOLO-OCR/dataset/52/train_52",
        'S': "C:/YOLO-OCR/dataset/53/train_53",
        'T': "C:/YOLO-OCR/dataset/54/train_54",
        'U': "C:/YOLO-OCR/dataset/55/train_55",
        'V': "C:/YOLO-OCR/dataset/56/train_56",
        'W': "C:/YOLO-OCR/dataset/57/train_57",
        'X': "C:/YOLO-OCR/dataset/58/train_58",
        'Y': "C:/YOLO-OCR/dataset/59/train_59",
        'Z': "C:/YOLO-OCR/dataset/5a/train_5a"
    }

    file_names = {}
    for char in char_list:
        file_names[char] = glob.glob(image_path[char] + "/*.png")

    return file_names


__file_names = {}
def sample_a_char_img(char):
    '''
    sample a character image
    char: a character

    return: a PIL typed image

    '''
    global __file_names

    if (0 == len(__file_names)):
        __file_names = load_image_file_names(["A", "B", "C", "D", "E", "F", "G", "H", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

    count = len(__file_names[char])
    if (count > 0):
        index = np.random.randint(low = 0, high = count)
        filename = __file_names[char][index]
        image = Image.open(filename)
        image = ImageOps.invert(image)
        image = image.convert("L")

        box = image.getbbox()
        pad_w = int((box[2] - box[0]) * 0.30 / 2.0 + 0.5)
        pad_h = int((box[3] - box[1]) * 0.30 / 2.0 + 0.5)
        left = box[0] - pad_w
        if left < 0:
            left = 0
        top = box[1] - pad_h
        if top < 0:
            top = 0
        right = box[2] + pad_w
        if right >= image.size[0]:
            right = image.size[0]
        bottom = box[3] + pad_h
        if bottom >= image.size[1]:
            bottom = image.size[1]
        box = (left, top, right, bottom)
        
        image = image.crop(box)
        #image = image.filter(ImageFilter.GaussianBlur(radius=1)) #(ImageFilter.MedianFilter)

        return image

    else:
        return None

def colorize_char_img(gray_img, Hrange, Srange, Vrange, varRange):
    '''
    transfer gray image into colorized image

    gray_img: input grayscale image
    Hrange, Srange, Vrange, varRange: color ranges

    return: colorized image
    '''

    _, color_map = gen_bg_img.gen_bg_img(Hrange, Srange, Vrange, varRange, gray_img.size)
    H, S, V = color_map.convert("HSV").split()
    arr_h = np.array(H).T
    arr_s = np.array(S).T
    arr_v = np.array(V).T
    k = np.array(gray_img)
    k = k / 255.0
    arr_h = arr_h * k
    arr_s = arr_s * k
    arr_v = arr_v * k

    arr_h = arr_h.astype(np.uint8)
    arr_s = arr_s.astype(np.uint8)
    arr_v = arr_v.astype(np.uint8)

    IH = Image.fromarray(arr_h, 'L')
    IS = Image.fromarray(arr_s, 'L')
    IV = Image.fromarray(arr_v, 'L')
    color_img = Image.merge("HSV", (IH, IS, IV)).convert("RGB")

    return color_img

def is_box_overlapped(box_list, box):
    '''
    is box overlapped ?
    '''
    for b in box_list:
        if (b[0] + b[2] < box[0]) or (box[0] + box[2] < b[0]) or \
           (b[1] + b[3] < box[1]) or (box[1] + box[3] < b[1]) :
            pass
        else:
            return True

    return False


def gen_dataset(main_file, img_path, img_count):
    '''
    '''
    char_list = ["A", "B", "C", "D", "E", "F", "G", "H"] #["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    fall = open(main_file, "w")
    for J in range(img_count):
        fall.write("data/obj/trainset/" + str(J).zfill(4) + ".png\r\n")

        _, back_img = gen_bg_img.gen_bg_img((0, 256), (0, 128), (127,256), (2, 15), (512, 512))
        
        color_img_list = []
        gray_img_list = []
        char_index = []

        max_char = np.random.randint(low = 10, high = 30)
        for i in range(max_char):
            index = np.random.randint(low = 0, high = len(char_list))
            char_index.append(index)
            char = char_list[index]
            gray_char_img = sample_a_char_img(char)
            color_char_img = colorize_char_img(gray_char_img, (0,256), (128, 250), (0, 128), (2, 15))
            gray_img_list.append(gray_char_img)
            color_img_list.append(color_char_img)
        
        fobj = open(img_path + str(J).zfill(4) + ".txt", "w")

        box_list = []
        i = 0
        for img in color_img_list:
            x = np.random.randint(low = 0, high= 480)
            y = np.random.randint(low = 0, high= 480)
            if ((x + img.size[0] < 512) and (y + img.size[1] < 512)):
                box = (x, y, img.size[0], img.size[1])
                if (not is_box_overlapped(box_list, box)):
                    back_img.paste(img, (x, y), gray_img_list[i])
                    box_list.append(box)
                    fobj.write("{0:d} {1:f} {2:f} {3:f} {4:f}\r\n".format(char_index[i], \
                        (x + img.size[0] / 2) / 512.0, (y + img.size[1] / 2) / 512.0, \
                        img.size[0] / 512.0, img.size[1] / 512.0))
            i += 1
        fobj.close()
        back_img = back_img.filter(ImageFilter.GaussianBlur(radius = 0.5 + 2 * np.random.random()))
        back_img.save(img_path + str(J).zfill(4) + ".png")
    fall.close()

def gen_trainset_validset():
    '''
    '''

    gen_dataset("C:/YOLO-OCR/train/data/train.txt", "C:/YOLO-OCR/train/data/obj/trainset/", 5000)
    gen_dataset("C:/YOLO-OCR/train/data/valid.txt", "C:/YOLO-OCR/train/data/obj/validset/", 200)

