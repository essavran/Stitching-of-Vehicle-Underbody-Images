import numpy as np
import cv2
import time, os
import matplotlib.pyplot as plt
from sklearn import metrics

# -*- coding: utf-8 -*- 

noice_matrix = np.zeros((2064, 30))
last_overlay = 0

def fill_noisy_matrix(selected_image):
    for x_position in range(500,1500):
        for y_position in range(12,18):
            is_black = selected_image[x_position][y_position][0] < 10 and selected_image[x_position][y_position][1] < 10 and selected_image[x_position][y_position][2] < 10
            if is_black:
                noice_matrix[x_position][y_position] += 1

def find_noisy_indices(noisy_count):
    constant = 150 
    indices = np.where(noice_matrix > noisy_count)
    if indices[0].size > 0:
        min = np.min(indices[0]) - constant
        max = np.max(indices[0]) + constant
        if min < 0:
            min = 0
        if max > 2063:
            max = 2063
        
        return min, max
    return 0, 0

def stitching(i, image1, image2, final_photo, noisy_min_x, noisy_max_x):
    # select 6 pixels from image2 -> 12-18
    start_x_image2 = 12
    end_x_image2 = 18
    x_interval = 6

    selected_region_image2 = image2[:, start_x_image2:end_x_image2]
    
    mutual_info_values =  []

    is_noisy = noisy_max_x > 0 and noisy_min_x > 0
    
    new_selected_image2 = []
    if is_noisy:
        new_selected_image2 =  np.concatenate((selected_region_image2[:noisy_min_x], selected_region_image2[noisy_max_x+1:]), axis=0)
    else:
        new_selected_image2 = selected_region_image2
    selected_image2_ravel = new_selected_image2.ravel()
    
    start_index_for_mi = 5
    end_index_for_mi = 25

    for x_position in range(start_index_for_mi,end_index_for_mi): 
        selected_region_image1 = image1[:, x_position:x_position+x_interval]
        
        new_selected_image1 = []
        if is_noisy:
            new_selected_image1 =  np.concatenate((selected_region_image1[:noisy_min_x], selected_region_image1[noisy_max_x+1:]), axis=0)
        else:
            new_selected_image1 = selected_region_image1
        
        hist_2d, _, _ = np.histogram2d(new_selected_image1.ravel(), selected_image2_ravel, bins=20)

        mi = metrics.mutual_info_score(None, None, contingency=hist_2d)
        mutual_info_values.append(mi)
            
    max_mi_index = np.argmax(mutual_info_values) + start_index_for_mi
    
    crop_x_image2 = 0

    if max_mi_index >= 23:
        crop_x_image2 = end_x_image2
    
    elif max_mi_index < start_x_image2: # back
        return final_photo, False, i
    
    elif max_mi_index + x_interval <= 30:
        crop_x_image2 = end_x_image2 + 30 - (max_mi_index + x_interval) 
    
    else:
        return final_photo, False, i 
    
    
    same_as_fisrt = image2[:, :crop_x_image2]
    image2 = image2[:, crop_x_image2:]

    if i != 0 and final_photo is not None : 
        image1 = final_photo
    if is_noisy:
        overlay = (crop_x_image2 - end_x_image2)
        
        image1_y_size = len(image1[0]) - 1
        same_as_fisrt_len = len(same_as_fisrt[0])-1
        
        const = 11
       
        global last_overlay
        total_overlay = overlay + last_overlay
        for i in range(noisy_min_x, noisy_max_x):
            for j in range(total_overlay):

                if same_as_fisrt[i][same_as_fisrt_len-const-j][0] > image1[i][image1_y_size-const-j][0] \
                and same_as_fisrt[i][same_as_fisrt_len-const-j][1] > image1[i][image1_y_size-const-j][1] \
                and same_as_fisrt[i][same_as_fisrt_len-const-j][2] > image1[i][image1_y_size-const-j][2]:

                    image1[i][image1_y_size-const-j] = same_as_fisrt[i][same_as_fisrt_len-const-j]
        last_overlay = overlay
    final_photo = cv2.hconcat([image1, image2])
    return final_photo, True, i

################################################

def main_stitching(folder_path):
    start_time = time.time()

    image_files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, img) for img in image_files]
    temp = folder_path.split("/")
    photoName = temp[2]

    image1 = None 
    image2 = None
    final_photo = None
    similarity = True

    image_size = len(image_paths)
    first15 = int(image_size * 15 / 100)
    last5 = int(image_size - (image_size * 5 / 100))

    # create noisy matrix & read images to an array
    all_images = np.empty(image_size, dtype = object)
    for i in range(first15, last5+1):
        image_i = cv2.imread(image_paths[i])
        all_images[i] = (cv2.rotate(image_i, cv2.ROTATE_90_COUNTERCLOCKWISE))
        fill_noisy_matrix(all_images[i])

    noisy_count = image_size*0.2 
    min_x_nosiy, max_x_noisy = find_noisy_indices(noisy_count)
    
    # stitching
    for i in range(first15, last5):
        if i == first15:
            image1 = all_images[i]
            image2 = all_images[i+1]
            imageIndex = i

        else:
            if similarity:
                all_images[i] = None 
                image1 = image2  
                image2 = all_images[i+1]
                imageIndex = i
            else:
                image2 = all_images[i+1]

        final_photo, similarity, imageIndex = stitching(imageIndex, image1, image2, final_photo, min_x_nosiy, max_x_noisy)

    if final_photo is not None:
        components = folder_path.split("/")
        result = "/".join(components[:4])
        print(result)
        if not os.path.exists(result+"/results"):
            os.makedirs(result+"/results")
        cv2.imwrite(result+f"/results/stitched_{components[5]}.jpg", final_photo)

    noice_matrix = np.zeros((2064, 30))

    end_time = time.time()
    elapsed_time = end_time - start_time

    return True, elapsed_time, result+f"/results/stitched_{components[5]}.jpg"