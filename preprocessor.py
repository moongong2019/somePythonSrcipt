import cv2
import numpy as np

H = 84
W = 84
C = 4
SPEED_LENGTH = 50
MIN_SPEED = 100

number_template = np.array( [[[  2,   1, 127, 253, 255, 251, 255, 254, 255,  23],
                              [  0,   0, 128, 255,  27,   0,   1, 128, 252, 255],
                              [  3,  77, 255, 254,  27,   0,   0, 129, 255, 252],
                              [  0,  76, 255,  72,   0,   1,   0, 128, 255, 255],
                              [  2,  78, 251,  79,   0,   2,   0, 128, 253,  24],
                              [ 24, 255, 255,  74,   1,   0,  80, 254, 255,  29],
                              [ 28, 252, 255,  76,   0,   3,  72, 255, 254,  24],
                              [ 24, 255, 127,   0,   0,   0,  78, 255, 255,  26],
                              [ 26, 255, 126,   0,   2,   0,  76, 255, 255,  25],
                              [255, 251, 129,   1,   2,   0,  77, 254, 255,  27],
                              [254, 255, 128,   0,   0,   0,  81, 255, 255,  28],
                              [255, 255, 124,   5,   3,   2,  71, 254,  76,   2],
                              [255, 253, 128,   0,   0,   0,  80, 255,  76,   1],
                              [255, 255, 130,   0,   0,  29, 255, 255,  74,   1],
                              [ 24, 255, 254, 255, 255, 252, 181,   0,   1,   1]],

                             [[  0,   3,   0,  63, 192, 255, 253, 255, 255, 255],
                              [  0,  64, 191, 254, 253, 255, 255, 255, 255, 253],
                              [255, 253, 255, 255, 255, 253, 255, 254, 255, 253],
                              [  1,   0,   0,  62, 192, 254, 252, 255, 254, 255],
                              [  0,   4,   3,  62, 191, 253, 255, 253, 255, 250],
                              [  2,   0,   0,  67, 191, 255, 255, 190,  64,   5],
                              [  0,  66, 193, 255, 253, 254, 255, 193,  66,   0],
                              [  0,  65, 188, 255, 255, 254, 255, 188,  63,   0],
                              [  2,  63, 192, 255, 254, 255, 255, 190,  64,   2],
                              [  0,  65, 193, 251, 255, 191,  65,   2,   3,   0],
                              [  0,  64, 191, 255, 255, 191,  59,   1,   0,   3],
                              [255, 254, 255, 255, 250, 193,  68,   0,   2,   0],
                              [255, 253, 255, 254, 255, 192,  61,   0,   0,   0],
                              [255, 255, 255, 255, 255, 189,  65,   0,   3,   0],
                              [253, 255, 255, 188,  64,   0,   2,   0,   0,   0]],

                             [[  0,   2, 129, 253, 254, 255, 253, 255, 255,  24],
                              [  1,  75, 126,   5,   0,   0,   1, 128, 253, 255],
                              [  1,  78, 129,   0,   2,   1,   1, 126, 254, 254],
                              [  1,   0,   0,   2,   0,   0,   0, 131, 255, 255],
                              [  0,   2,   2,   0,   1,   2,   1, 127, 252,  25],
                              [  0,   0,   1,   0,   0,   0,   0, 128, 255,  30],
                              [  2,   0,   0,   0,   3,   3,  75, 255, 255,  25],
                              [  0,   1,   0,   2,   0,   0,  75, 255,  76,   0],
                              [  1,   0,   2,   0,   1, 255, 254, 130,   0,   0],
                              [  0,   3,   0,   1, 230, 228,   0,   0,   0,   0],
                              [  3,   0,   0, 177, 255,   1,   2,   1,   0,   0],
                              [  0,   1, 128, 255,  26,   0,   1,   0,   0,   0],
                              [  0,  79, 127,   0,   2,   0,   0,   3,   0,   0],
                              [255, 251, 131,   0,   1,   1,   1,   0,   0,   0],
                              [253, 255, 255, 254, 255, 254, 254, 127,   0,   0]],

                             [[  1,   0, 130, 254, 254, 254, 255, 253, 253, 255],
                              [  0,   1,   0,   2,   2,   4,  73, 255, 255,  25],
                              [  3,   3,   0,   1,   0,   0,  78, 255,  76,   1],
                              [  0,   0,   0,   2,   1, 254, 255, 130,   2,   0],
                              [  0,   2,   0,   1, 228, 255, 253, 126,   0,   2],
                              [  0,   3,   0, 177, 255, 254, 253, 255, 254,  27],
                              [  0,   1,   0,   0,   1,   0,   0, 130, 255,  25],
                              [  0,   0,   2,   0,   0,   2,   0, 126, 255, 255],
                              [  0,   3,   1,   0,   1,   0,   1, 128, 254, 255],
                              [  1,   0,   1,   1,   4,   0,   0, 129, 255,  25],
                              [  0,   1,   0,   0,   0,   4,  72, 255, 255,  28],
                              [  1,   0,   2,   4,   1,   0,  80, 253, 252,  28],
                              [  0,   1,   0,   0,   0,   0,  73, 255, 255,  27],
                              [229,   3,   0,   0,   1,   0,  75, 254,  76,   0],
                              [ 26, 251, 255, 254, 255, 255, 255, 128,   0,   0]],

                             [[  0,   0,   0,   2,   0,   1,   0, 193, 255, 255],
                              [  2,   0,   1,   0,   1,   0, 165, 255, 254, 255],
                              [  0,   5,   0,   1,   0, 140, 255, 255, 255, 255],
                              [  2,   0,   0,   2, 114, 255, 249, 255, 254, 255],
                              [  0,   6,  63, 254, 255, 114,   1, 194, 255, 254],
                              [  1,  36, 254, 166,   0,   1,   1, 189, 255, 254],
                              [  2,  35, 194,   0,   0,   1,   0, 192, 254, 255],
                              [253, 219,   0,   0,   2,   0,   0, 192, 255,  13],
                              [253, 220,   0,   0,   4,   0, 166, 255, 255,  13],
                              [255, 254, 255, 255, 251, 255, 251, 255, 255, 251],
                              [  0,   0,   0,   0,   4,   3, 166, 255, 254,  15],
                              [  0,   0,   5,   0,   0,   0, 165, 253,  41,   0],
                              [  1,   2,   0,   0,   1,   0, 167, 255,  38,   0],
                              [  0,   1,   2,   0,   1, 138, 255, 254,  37,   2],
                              [  3,   0,   2,   0,   0, 141, 253, 255,  40,   0]],

                             [[  0,   0, 127, 255, 255, 254, 255, 255, 253, 255],
                              [  0,  77, 255, 255,  22,   2,   1,   0,   1,   0],
                              [  2,  72, 255,  74,   2,   0,   0,   2,   0,   0],
                              [  0,  81, 254,  74,   0,   2,   0,   0,   2,   0],
                              [  0,  75, 254,  77,   3,   0,   1,   1,   0,   4],
                              [ 29, 252, 255, 255, 251, 255, 253, 254,  78,   0],
                              [  0,   0,   0,   0,   1,   1,  80, 253, 254,  26],
                              [  1,   0,   1,   0,   2,   0,  76, 255, 255,  26],
                              [  0,   1,   0,   1,   0,   0,  78, 255, 255,  25],
                              [  0,   1,   0,   0,   2,   1,  75, 253, 255,  27],
                              [  3,   0,   1,   0,   0,   2,  77, 255, 255,  28],
                              [  2,   0,   0,   0,   0,   0,  76, 255,  76,   2],
                              [  0,   2,   0,   5,   1,   0,  80, 254,  76,   1],
                              [231,   0,   2,   0,   0,   0,  71, 255,  74,   1],
                              [254, 255, 251, 255, 254, 255, 255, 126,   1,   1]],

                             [[  0,   0,   3,   0,   0, 255,   2,   0, 255, 255],
                              [  2,   0,   4,   0, 255, 254,   0,   2,   0,   0],
                              [  1,   0,   0, 255, 255,   2,   0,   0,   0,   0],
                              [  0,   1,   0, 254,   0,   0,   4,   0,   0,   0],
                              [  1, 253, 255, 255,   0,   0,   0,   1,   0,   0],
                              [  0, 255, 255, 253, 255, 255, 254, 255, 255, 255],
                              [  2, 255, 254, 255,   0,   0,   0,   1, 255, 255],
                              [  0, 255, 255,   0,   3,   0,   1,   0, 255, 255],
                              [255, 255, 254,   2,   1,   0,   0, 255, 255, 253],
                              [255, 255, 255,   0,   0,   3,   0, 252, 254, 255],
                              [255, 254, 253,   1,   1,   0,   3, 255, 255, 254],
                              [254, 255,   0,   5,   0,   2,   0, 254, 253, 255],
                              [255, 253,   0,   0,   1,   0,   0, 255, 255, 255],
                              [254, 255,   1,   0,   2,   2,   0, 255, 254,   3],
                              [  2,   2, 253, 255, 253, 251, 255, 252,   0,   0]],

                             [[255, 255, 255, 251, 255, 255, 254, 255, 255, 255],
                              [  0,   0,   1,   6,   0,   0,   3, 253, 254, 255],
                              [  0,   0,   0,   0,   1,   0,   2, 255, 255, 254],
                              [  3,   0,   2,   0,   0,   4,   1, 253, 255, 254],
                              [  0,   0,   0,   0,   1,   0,   0, 255, 255,   0],
                              [  1,   0,   2,   1,   0,   1, 255, 252, 254,   0],
                              [  1,   0,   2,   0,   0,   0, 254,   2,   0,   0],
                              [  0,   2,   0,   1,   0, 255,   0,   0,   0,   2],
                              [  0,   1, 253, 255, 255,   0,   0,   1,   0,   0],
                              [  1,   0, 255, 253, 255,   2,   3,   0,   0,   0],
                              [  0,   1, 255, 252, 255,   1,   0,   0,   0,   0],
                              [  1,   3, 252, 255, 255,   0,   0,   0,   0,   0],
                              [  0,   0, 255, 253,   0,   0,   4,   1,   0,   0],
                              [  1, 255, 255, 255,   2,   0,   0,   0,   0,   0],
                              [  0, 253, 254,   0,   0,   2,   3,   0,   0,   0]],

                             [[  1,   0, 133, 255, 252, 255, 254, 255, 254,  28],
                              [  0,  77, 249, 255,  29,   1,   0, 126, 255, 255],
                              [  0,  79, 255,  75,   1,   0,   0, 131, 255, 255],
                              [  0,  76, 254,  75,   1,   0,   1, 125, 255,  23],
                              [ 27, 255, 254,  75,   2,   0,  77, 255, 255,  29],
                              [ 24, 255, 253,  77,   0,   2,  74, 255, 253,  24],
                              [  0,  77, 255,  76,   0,   0,  79, 252,  78,   0],
                              [  1,  76, 254, 255, 255, 253, 254, 255,  74,   4],
                              [  0,  76, 255,  76,   0,  24, 255, 255,  75,   3],
                              [ 27, 255, 126,   1,   1,   0,  76, 255, 255,  24],
                              [253, 255, 128,   0,   0,   2,  74, 253,  76,   1],
                              [255, 252, 132,   0,   1,   0,  74, 255,  78,   0],
                              [254, 252, 129,   0,   0,  26, 255, 252,  75,   0],
                              [255, 255, 126,   1,   0,  27, 252, 255,  78,   0],
                              [254, 252, 255, 254, 255, 253, 255, 128,   0,   3]],

                             [[  2,  38, 255, 251, 255, 255, 255, 255, 255, 255],
                              [  0,  40, 253, 168,   5,   0,   1, 191, 255, 255],
                              [  2,  39, 254, 166,   0,   1,   0, 194, 255, 255],
                              [ 12, 251, 255, 164,   1,   1,   0, 189, 255, 255],
                              [ 14, 255, 255, 166,   0,   1,   0, 192, 255, 255],
                              [ 13, 253, 192,   0,   2,   0,   0, 188, 255, 255],
                              [ 11, 255, 192,   0,   1,   1,   0, 196, 255, 255],
                              [ 16, 250, 193,   0,   0,   1,   1, 189, 255, 255],
                              [255, 254, 192,   1,   0,   1,   0, 191, 255,  10],
                              [ 14, 254, 255, 253, 255, 255, 251, 255, 253,  16],
                              [  0,   2,   1,   0,   0,   0,   2, 189, 255,  9],
                              [  0,   0,   0,   3,   0,   0,   0, 189, 255,  14],
                              [  4,   0,   4,   0,   1,   1, 167, 255,  38,   2],
                              [  0,   2,   0,   1, 116, 251,  89,   0,   1,   0],
                              [  1,   2,  60, 252, 255, 118,   0,   1,   1,   0]]])

class Number_OCR():
    def __init__(self, display_shape):
        self.display_shape = display_shape
        # Get KNN
        self.knn = cv2.ml.KNearest_create()
        train_label = [0,1,2,3,4,5,6,7,8,9]
        train_label = np.array(train_label).reshape(-1,1)
        train_data = np.array(number_template).reshape(10, 150).astype(np.float32)
        self.knn.train(train_data,cv2.ml.ROW_SAMPLE,train_label)

    # ocr
    def predict(self, full_img):
        if self.display_shape[1] == '2280':
            height_start, height_end = 477, 503
            width_start, width_end = 510, 552
        elif self.display_shape[1]==1280:
            #height_start, height_end = 478, 503
            #width_start, width_end = 420, 459
            height_start, height_end = 333, 357 #Y����
            width_start, width_end = 290, 322 #X����
        else:  #1920 x 1080
            height_start, height_end = 493, 530  #minicap
            #height_start, height_end = 500, 535  #摄像头
            width_start, width_end = 430, 484
            
        sub_img = full_img[height_start:height_end, width_start:width_end, :]
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

        sub_img = ((sub_img>190) * 255).astype(np.uint8)
		#sub_img = ((sub_img > 210) * 255).astype(np.uint8)
        #cv2.imshow('tmp3', sub_img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
        sub_img = cv2.morphologyEx(sub_img, cv2.MORPH_CLOSE, kernel)
        _, contours, _ = cv2.findContours(sub_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        number_list = []
        position_list = []
        for cnt in contours:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if w*h > 10:
                number_list.append(cv2.resize(sub_img[y:y+h, x:x+w], (10,15)))
                position_list.append(x)

        temp = [(number_array, x) for number_array, x in zip(number_list, position_list)]
        temp = sorted(temp, key = lambda x: x[1])
        number_list = np.array([number_array for number_array, x in temp]).reshape(-1, 150).astype(np.float32)

        if len(number_list)>0:
            _, results, _, _ = self.knn.findNearest(number_list, 1)

            number = 0

            for r in results:
                number = 10*number + int(r)

            number = number%400

            return number, sub_img
        else:
            return 0, None

class Preprocessor():
    def __init__(self, display_shape):
        self.display_shape = display_shape
        self.ocr = Number_OCR(display_shape)
        self.last_4_images = [np.zeros((H, W), np.uint8) for _ in range(3)]
        self.speed_list = []
        self.threshold = 180

    def get_gray_img(self, img):
        sub_img = img[65:170, 805: 940]  #minicap
        #sub_img = img[92:190, 810: 940]   #摄像头
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        sub_img = cv2.resize(sub_img, (H, W))
        #sub_img = self.get_feature_img(sub_img)
        #sub_img = ((sub_img > 150) * 255).astype(np.uint8)
        return sub_img
        
    def get_feature_img(self,img):
        # img = cv2.resize(img, (64, 32), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=-1)
        height, width, depth = np.array(img).shape
        black = [0, 0, 0]
        # print depth
        # 将非白区域变白
        for i in range(height):
            for j in range(width):
                if img[i, j, :][0] > self.threshold and img[i, j, :][1] > self.threshold and img[i, j, :][2] > self.threshold:
                    # if white in img[i, j, :]:
                    pass
                else:
                    img[i, j, :] = black
                    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def process(self, img):

        speed, _ = self.ocr.predict(img)
        sub_img = self.get_gray_img(img)
        self.last_4_images.append(sub_img)
        self.speed_list.append(speed)

        if len(self.last_4_images) > C:
            self.last_4_images.pop(0)
        if len(self.speed_list) > SPEED_LENGTH:
            self.speed_list.pop(0)

        input_one_record = np.dstack(
        (
            self.last_4_images[-4].ravel(),
            self.last_4_images[-3].ravel(),
            self.last_4_images[-2].ravel(),
            self.last_4_images[-1].ravel()
        )
        ).ravel().reshape(H, W, C)

        if np.mean(self.speed_list) < MIN_SPEED and len(self.speed_list)>SPEED_LENGTH -5:
            print('++++++++++++RESET++++++++++++++++++')
            self.speed_list = []
            reset_flag = True
        else:
            reset_flag = False

        return input_one_record, speed, reset_flag
"""
np.set_printoptions(threshold=np.nan)
number_template_list=[]
for i in range(10):
    number_template_list.append(cv2.imread('number_template/%d.jpg'%i, cv2.IMREAD_GRAYSCALE))
number_template = np.array(number_template_list)
print(number_template)
"""
def main():
    image = cv2.imread('0010991.jpg')
    p = Preprocessor(display_shape=(960,1080))
    p.get_gray_img(image)
    #ocr = Number_OCR((1080, 1960))
    #print(ocr.predict(image))
    cv2.imshow('tmp',image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()