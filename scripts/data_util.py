import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image_file_paths(filepath='../dataset/driving_log.csv'):
  '''
  Loads lines from the CSV log file containing information about image-file-paths.
  '''
  lines = []
  with open(filepath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  return lines

def get_random_image(image_set_path='../dataset'):
  '''
  Returns a random image from the dataset.
  '''
  lines = load_image_file_paths()
  rand_img_idx = np.random.randint(0, len(lines))
  img_path = (image_set_path + '/IMG/' + lines[rand_img_idx][0].split('/')[-1])
  return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def display_flipped_images(image_set_path='../dataset'):
  '''
  Displays flipped images.
  '''
  img = get_random_image(image_set_path)
  flipped_img = cv2.flip(img, 1)
  fig, axs = plt.subplots(1, 2)
  axs[0].axis('off')
  axs[0].set_title('Source image')
  axs[0].imshow(img)
  axs[1].axis('off')
  axs[1].set_title('Flipped image')
  axs[1].imshow(flipped_img)
  plt.show()

def display_cropped_images(image_set_path='../dataset'):
  '''
  Displays a random image from the dataset and its cropped image.
  '''
  img = get_random_image(image_set_path)
  height,width = img.shape[:2]
  cropped_img = img[70:height-25,]
  fig,axs = plt.subplots(1, 2)
  axs[0].axis('off')
  axs[0].set_title('Source image')
  axs[0].imshow(img)
  axs[1].axis('off')
  axs[1].set_title('Cropped image')
  axs[1].imshow(cropped_img)
  plt.show()

