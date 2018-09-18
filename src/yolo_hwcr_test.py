from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import yolo_hwcr_model

def img2feat(image_path):
    '''
    input a image, output a yolo feature vector
    '''

    #prepare input X
    image = Image.open(image_path)
    pixels = np.array(image)
    data = pixels / 255.0
    X = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])

    #prepare model
    model = yolo_hwcr_model.create_model((None, None, 3))
    model.load_weights('../model_data/yolo_hwcr.h5')

    #do inference
    Y = model.predict(X)

    return Y

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def feat2box(feat, thresh, img_size, anchor_size):
    '''
    input feature vector , output box list
    '''
    
    grid_num_y = feat.shape[1]
    grid_num_x = feat.shape[2]

    grid_size_x = img_size[0] / grid_num_x
    grid_size_y = img_size[1] / grid_num_y

    cy = np.arange(grid_num_y) * grid_size_y
    cx = np.arange(grid_num_x) * grid_size_x

    #be causious about the x-y order!
    bxy = np.array([(y, x) for y in cy for x in cx]).reshape(grid_num_y, grid_num_x, -1)
    bxy += sigmoid(feat[0, ..., 1::-1]) * np.array([grid_size_x, grid_size_y])

    bw = anchor_size[0] * np.exp(feat[0, ..., 2:3])
    bh = anchor_size[1] * np.exp(feat[0, ..., 3:4])
    box_conf = sigmoid(feat[0, ..., 4])
    cls_prob = sigmoid(feat[0, ..., 5:13])

    filtering = box_conf > thresh

    bxy = bxy[filtering]
    bw = bw[filtering]
    bh = bh[filtering]
    box_conf = box_conf[filtering]
    cls_prob = cls_prob[filtering]
    cls_id = np.argmax(cls_prob, axis = 1)

    box_list = []
    for i in range(len(bxy)):
        box = (bxy[i], bw[i], bh[i], cls_prob[i][cls_id[i]] * box_conf[i], cls_id[i])
        box_list.append(box)

    return box_list

def draw_box_on_image(image_path, box_list):
    '''
    draw boxes on image
    '''
    char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    image = Image.open(image_path)
    drawer = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size = 16)

    for i in range(len(box_list)):
        box = box_list[i]
        w = box[1]
        h = box[2]
        x = box[0][1] - w / 2
        y = box[0][0] - h / 2
        p = box[3]
        c = char_list[box[4]]
        drawer.text((x, y - 16), c + '=' + format(p, '.2f'), font = font, fill='#ff0000')
        drawer.rectangle([(x,y), (x + w, y + h)], outline='#ff0000')

    image.show()

def test(image_path):
    '''
    test
    '''

    feat = img2feat(image_path)
    box_list = feat2box(feat, 0.7, (512, 512), (55.8024,60.7835))
    draw_box_on_image(image_path, box_list)


