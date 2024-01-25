import cv2
import numpy as np
import cv2
import os
import time

# sift + brute force
def find_matching_points(i, img1, img2, final_photo):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None) # keypoints, descriptors
    kp2, des2 = sift.detectAndCompute(img2, None)

    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    matching_points = []  

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

    totalSum = 0
    count = 0
    countBack = 0

    for match in matching_points:
        (x1, y1), (x2, y2) = match
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        #print(f"x1 = {x1} x2 = {x2} y1 = {y1} y2 = {y2}")
        if abs(y1 - y2) < 10:
            if x1 - x2 > 0:
                totalSum += abs(30 - x1 + x2) 
                count += 1
            else:
                countBack += 1
    
    if countBack >= count:
        return final_photo, True, i
    
    if count == 0:
        return final_photo, False, i

    if count > 6000:
        return final_photo, True, i
    
    if count != 0 : 
        average = totalSum / count
        x_crop = round(average) 
        
        if i != 0 and final_photo is not None :
            img1 = final_photo

        if x_crop >= 30: 
            return final_photo, False, i
        
        img2 = img2[:, x_crop:]
        
        shape = img1.shape
        # if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
        #     shape2 = img2.shape
        #     if len(shape2) == 3 and shape2[0] > 0 and shape2[1] > 0 and shape2[2] > 0:
        #         if genislik != 0:
        #             final_photo = cv2.hconcat([img1, img2])
        #             return final_photo
        final_photo = cv2.hconcat([img1, img2])
        return final_photo, True, i

    else:
        return final_photo, False, i


#################################################
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
        similarity = True

        for i in range(len(image_paths) - 1):    
            if i == 0:
                img1 = cv2.imread(image_paths[i])
                img2 = cv2.imread(image_paths[i+1])
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                imageIndex = i

            else:
                if similarity:
                    img1 = img2
                    img2 = cv2.imread(image_paths[i+1])
                    img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    imageIndex = i
                else:
                    img2 = cv2.imread(image_paths[i+1])
                    img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

            final_photo, similarity, imageIndex = find_matching_points(imageIndex, img1, img2, final_photo)
            
        if final_photo is not None:
            print(folder_path)
            cv2.imwrite(f"sift_results_final/SIFT_final_photo_{photoName}.jpg", final_photo)