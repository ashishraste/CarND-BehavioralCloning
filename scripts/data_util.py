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

def display_flipped_images(image_set_path='../dataset'):
  '''
  Displays flipped images.
  '''
  lines = load_image_file_paths()
  rand_img_idx = np.random.randint(0, len(lines))
  img_path = (image_set_path + '/IMG/' + lines[rand_img_idx][0].split('/')[-1])
  img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
  flipped_img = cv2.flip(img, 1)
  fig, axs = plt.subplots(1, 2)
  axs[0].axis('off')
  axs[0].set_title('Source image')
  axs[0].imshow(img)
  axs[1].axis('off')
  axs[1].set_title('Flipped image')
  axs[1].imshow(flipped_img)
  plt.show()


