import cv2
import numpy as np
import cv2
import os
import time

# sift + brute force
def find_matching_points(i, nameIndex, img1, img2, final_photo, photoIndex):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None) # keypoints, descriptors
    kp2, des2 = sift.detectAndCompute(img2, None)

    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    start_time = time.time()
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    matching_points = []  

    start_time = time.time()

    try:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                img1_idx = m.queryIdx
                img2_idx = m.trainIdx
                x1, y1 = kp1[img1_idx].pt
                x2, y2 = kp2[img2_idx].pt
                matching_points.append(((x1, y1), (x2, y2)))
    except ValueError as e:
        print("error - matching points ")
  
    start_time = time.time()
    totalSum = 0
    count = 0

    for match in matching_points:
        (x1, y1), (x2, y2) = match
        if abs(y1 - y2) < 20: 
            totalSum += abs(30 - x1 + x2) 
            count += 1

    if count != 0 : 
        start_time = time.time()

        average = totalSum / count
        x_crop = int(average) 
        
        if i == 0 or final_photo is None :
            img1_new = img1
        else: 
            img1_new = final_photo
        img2_new = img2

        if i == 0:
            img1_new_rotate = img1_new
        else:
            img1_new_rotate = img1_new
        img2_new_rotate = img2

        yukseklik, genislik,y = img1_new_rotate.shape

        if x_crop == 30: 
            return -1, final_photo
        img2_new_crop = img2_new_rotate[:, x_crop:]
        
        yukseklik, genislik,y = img2_new_crop.shape

        shape = img1_new_rotate.shape
        if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            shape2 = img2_new_crop.shape
            if len(shape2) == 3 and shape2[0] > 0 and shape2[1] > 0 and shape2[2] > 0:
                if genislik != 0:
                    final_photo = cv2.hconcat([img1_new_rotate, img2_new_crop])
                    return -1, final_photo
            else:
                return -1, final_photo
        else:
            return -1, final_photo
        
    else:

        if i == 0 or final_photo is None :
            img1_new = img1
        else: 
            img1_new = final_photo
        img2_new = img2

        if i == 0:
            img1_new_rotate = img1_new
        else:
            img1_new_rotate = img1_new
        img2_new_rotate =  img2_new
        shape = img1_new_rotate.shape
        yukseklik, genislik,y = img2_new_rotate.shape
        if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            shape2 = img2_new_rotate.shape
            if len(shape2) == 3 and shape2[0] > 0 and shape2[1] > 0 and shape2[2] > 0:
                if genislik != 0:
                    final_photo = cv2.hconcat([img1_new_rotate, img2_new_rotate])
                
        if photoIndex == -1:
            return i, final_photo 
        else:
            return photoIndex, final_photo


############################33
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

        img1 = None 
        img2 = None
        final_photo = None
        photoIndex = -1
        for i in range(len(image_paths) - 1):
            start_time = time.time()
            if i == 0:
                img1 = cv2.imread(image_paths[i])
                img2 = cv2.imread(image_paths[i+1])
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                if photoIndex == -1:
                    img1 = img2
                else:
                    img1 = cv2.imread(image_paths[photoIndex])

                img2 = cv2.imread(image_paths[i+1])
                img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

            nameIndex = 32
            if photoIndex == -1:
                photoIndex, final_photo = find_matching_points(i, folder_path, img1, img2, final_photo, photoIndex)
            else: 
                photoIndex, final_photo = find_matching_points(i, folder_path, img1, img2, final_photo, photoIndex)
            
        if final_photo is not None:
            print(folder_path)
            cv2.imwrite(f"final_photo_20_{photoName}.jpg", final_photo)