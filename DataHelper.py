import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DataHelper():
  @staticmethod
  def load_data(folder,image_shape,image_type,flip_lr=True,load_n_percent=100):
    img_rows,img_cols,channels = image_shape
    glob_glob = folder + "/*" + image_type
    images = glob.glob(glob_glob)
    print("LOADING FROM %s"%(glob_glob))
    print("LOADING %d IMAGES"%len(images))
    x = []
    num_images = len(images)
    for n,i in enumerate(images):
      if 100*n/num_images == load_n_percent:
        break
      img = Image.open(i)
      if channels == 4:
        img = img.convert('RGBA')
      elif channels == 3:
        img = img.convert('RGB')
      elif channels == 1:
        img = img.convert('L')
      img = img.resize(size=(img_rows,img_cols),resample=Image.ANTIALIAS)
      img = np.array(img).astype('float32')
      img = img/255
      x.append(img)
      if flip_lr:
        x.append(np.fliplr(img))
      
    print("LOADED %d IMAGES"%len(x))
    i = np.random.randint(0,len(x)-1,size=1)[0]
    print("SHOWING IMAGE: ",i)
    displayed_img = x[i]
    img_min = displayed_img.min()
    img_max = displayed_img.max()
    displayed_img = (displayed_img-img_min)/(img_max - img_min)
    plt.imshow(displayed_img)
    return x

  @staticmethod
  def save_images(epoch,generated_images,img_shape,num_rows,num_cols,preview_margin,output_path,image_type):
    image_count = 0
    img_size = img_shape[1]
    channels = img_shape[-1]
    preview_height = num_rows*img_size + (num_cols + 1)*preview_margin
    preview_cols = num_cols*img_size + (num_cols + 1)*preview_margin
    image_array = np.full((preview_height, preview_cols, channels), 255, dtype=np.uint8)
    for row in range(num_rows):
      for col in range(num_cols):
        r = row * (img_size+preview_margin) + preview_margin
        c = col * (img_size+preview_margin) + preview_margin
        img = generated_images[image_count]
        img_min = img.min()
        img_max = img.max()
        image_array[r:r+img_size,c:c+img_size] = 255*(img - img_min)/(img_max-img_min)
        image_count += 1
  
    filename = os.path.join(output_path,f"train-{epoch}" + image_type)
    if channels == 1:
      im = Image.fromarray(image_array[0],mode='L')
    else:
      im = Image.fromarray(image_array)
    im.save(filename)
