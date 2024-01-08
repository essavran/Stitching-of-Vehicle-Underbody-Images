import numpy as np
from sklearn import metrics
import cv2
import time, os

def stitching(i, image1, image2, final_photo):
    end_x_image1 = image1.shape[1] - 1
    selected_region_image1 = image1[:, end_x_image1:end_x_image1+1]
    y_selected_region_image1, x_selected_region_image1, _ = selected_region_image1.shape

    y_image2, x_image2, _ = image2.shape

    mutual_info_values = []
    for x_position in range(x_image2):
        selected_region_image2 = image2[:, x_position:x_position+1]
        y_image2, x_image2, _ = selected_region_image2.shape

        hist_2d, _, _ = np.histogram2d(selected_region_image1.flatten(), selected_region_image2.flatten(), bins=100)
    
        mi = metrics.mutual_info_score(None, None, contingency=hist_2d)
        mutual_info_values.append(mi)

    max_mi_index = np.argmax(mutual_info_values)
    max_mi_value = np.max(mutual_info_values)

    # print(f"Maximum Mutual Information Value: {max_mi_value}")
    # print(f"Corresponding x value: {max_mi_index}")
    
    # if max_mi_index < 3:
    #     return final_photo, True, i
    
    # if max_mi_index > 28: 
    #     return final_photo, False, i
    
    # if max_mi_value <= 1.5:
    #     return final_photo, False, i
    
    # if i >= 1500:
    #     if max_mi_value < 2.20:
    #         return final_photo, False, i

    image2 = image2[:, max_mi_index:]

    if i != 0 and final_photo is not None : # TODO: check it
        image1 = final_photo
    final_photo = cv2.hconcat([image1, image2])
    return final_photo, True, i

################################################
source_folder = "stitching/stitching"
files = os.listdir(source_folder)
index = 0
for file_name in files:
    source_path = os.path.join(source_folder, file_name)
    if os.path.isdir(source_folder+'/'+file_name):
        folder_path = source_folder+'/'+file_name+'/Normal'
        print(f"{file_name} {folder_path}")
        image_files = sorted(os.listdir(folder_path))
        image_paths = [os.path.join(folder_path, img) for img in image_files]
        temp = folder_path.split("/")
        photoName = temp[2]

        image1 = None 
        image2 = None
        final_photo = None
        similarity = True
        start_time_total = time.time()

        for i in range(len(image_paths) - 1):
            start_time = time.time()
            if i == 0:
                image1 = cv2.imread(image_paths[i])
                image2 = cv2.imread(image_paths[i+1])
                image1 = cv2.rotate(image1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                image2 = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                imageIndex = i

            else:
                if similarity:
                    image1 = image2  
                    image2 = cv2.imread(image_paths[i+1])
                    image2 = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    imageIndex = i
                else:
                    image2 = cv2.imread(image_paths[i+1])
                    image2 = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)

            final_photo, similarity, imageIndex = stitching(imageIndex, image1, image2, final_photo)
            end_time = time.time()
            elapsed_time = end_time - start_time
            #print(f"tüm işlem - Işlem {elapsed_time} saniye sürdü.")

        if final_photo is not None:
            print(folder_path)
            cv2.imwrite(f"mi_results/MI_final_photo_22_{photoName}.jpg", final_photo)

        end_time_total = time.time()
        elapsed_time_total = end_time_total - start_time_total
        print(f"FINAL ---> Total time {elapsed_time_total}")