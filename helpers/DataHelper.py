from config.TrainingConfig import DataConfig
import os
import glob
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


class DataHelper(DataConfig):
    def __init__(self, data_config: DataConfig):
        super().__init__(**data_config.__dict__)
        self.image_output_path = self.data_path + "/images"

    def load_data(self):
        img_rows, img_cols, channels = self.image_shape
        glob_glob = self.data_path + "/*" + self.image_type
        images = glob.glob(glob_glob)
        print("LOADING FROM %s" % (glob_glob))
        print("LOADING %d IMAGES" % len(images))
        x = []
        num_images = len(images)
        for n, i in enumerate(images):
            if 100*n/num_images >= self.load_n_percent:
                break
            img = Image.open(i)
            if channels == 4:
                img = img.convert('RGBA')
            elif channels == 3:
                img = img.convert('RGB')
            elif channels == 1:
                img = img.convert('L')
                # 
                
            img = img.resize(size=(img_rows, img_cols),resample=Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = self.load_scale_function(img)
            # if channels == 1:
            #     img = np.expand_dims(img,axis=-1)
            x.append(img)
            if self.flip_lr:
                x.append(np.fliplr(img))

        print("LOADED %d IMAGES" % len(x))
        i = np.random.randint(0, len(x)-1, size=1)[0]
        print("SHOWING IMAGE: ", i)
        displayed_img = x[i]
        img_min = displayed_img.min()
        img_max = displayed_img.max()
        displayed_img = (displayed_img-img_min)/(img_max - img_min)
        plt.imshow(displayed_img)
        return x

    def save_images(self, epoch, generated_images,preview_rows,preview_cols,preview_margin):
        image_count = 0
        img_size = self.image_shape[1]
        channels = self.image_shape[-1]
        preview_height = preview_rows*img_size + (preview_rows + 1)*preview_margin
        preview_width = preview_cols*img_size + (preview_cols + 1)*preview_margin
        
        if channels ==1:
            image_array = np.full((preview_height, preview_width), 255, dtype=np.uint8)
        else:
            image_array = np.full((preview_height, preview_width, channels), 255, dtype=np.uint8)
        for row in range(preview_rows):
            for col in range(preview_cols):
                r = row * (img_size+preview_margin) + preview_margin
                c = col * (img_size+preview_margin) + preview_margin
                img = generated_images[image_count]
                if channels == 1:
                    img = np.reshape(img,newshape=(img_size,img_size))
                scaled_img = self.save_scale_function(img)
                image_array[r:r+img_size, c:c+img_size] = scaled_img
                image_count += 1

        filename = os.path.join(self.image_output_path, f"train-{epoch}" + self.image_type)
        im = Image.fromarray(image_array.astype(np.uint8))
        im.save(filename)
