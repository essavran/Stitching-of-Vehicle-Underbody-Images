import numpy as np
import cv2
import time, os

def calculate_mutual_information_histogram(x, y, bins=20):
    # 100, 20 , 6,  12 , 45, 16 
    hgram, xedges, yedges = np.histogram2d(x, y, bins=bins)#, density=True)

    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def stitching(i, image1, image2, final_photo):
    # select 6 pixels from image2 -> 12-18
    start_x_image2 = 12
    end_x_image2 = 18
    x_interval = 6

    selected_region_image2 = image2[:, start_x_image2:end_x_image2]
    
    # loop for image1
    y_image1, x_image1, _ = image1.shape
    mutual_info_values = []

    for x_position in range(x_image1-x_interval): 
        selected_region_image1 = image1[:, x_position:x_position+x_interval]
    
        mi = calculate_mutual_information_histogram(selected_region_image1.ravel(), selected_region_image2.ravel()) 
        # image2 için ravel döngü dışında yap
        mutual_info_values.append(mi)

    max_mi_index = np.argmax(mutual_info_values)
    max_mi_value = np.max(mutual_info_values)

    crop_x_image2 = 0

    #print(f"Maximum Mutual Information Value: {max_mi_value}")
    #print(f"Corresponding x value: {max_mi_index}")
    
    if max_mi_index >= 23: # 21
        crop_x_image2 = end_x_image2

    elif max_mi_index == start_x_image2:
        return final_photo, True, i # TODO: check
    
    elif max_mi_index < start_x_image2:
        return final_photo, False, i
    
    elif max_mi_index + x_interval <= 30:
        crop_x_image2 = end_x_image2 + 30 - (max_mi_index + x_interval) 

    image2 = image2[:, crop_x_image2:]

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

        image_size = len(image_paths)
        first15 = int(image_size * 15 / 100)
        last5 = int(image_size - (image_size * 5 / 100))
        print(f"{first15} and {last5}")
        #for i in range(len(image_paths) - 1):
        for i in range(first15, last5):
            start_time = time.time()
            if i == first15:
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
                    imageIndex = i

            final_photo, similarity, imageIndex = stitching(imageIndex, image1, image2, final_photo)
            end_time = time.time()
            elapsed_time = end_time - start_time

        if final_photo is not None:
            print(folder_path)
            cv2.imwrite(f"mi_results_v7_45/MI_final_photo_34_{photoName}.jpg", final_photo)

        end_time_total = time.time()
        elapsed_time_total = end_time_total - start_time_total
        print(f"FINAL ---> Total time: {elapsed_time_total}")