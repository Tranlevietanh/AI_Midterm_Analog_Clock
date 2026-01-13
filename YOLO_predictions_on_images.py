import os
import cv2
import time
from ultralytics import YOLO

model = YOLO(r"C:\Users\VBK computer\Downloads\best.pt")
CLOCK_CLASS_ID = 0 #clock ID là 0 cho mô hình customized, nhưng là 74 với mô hình "YOLO11n.pt" gốc

image_folder = r'D:\Bai tap\Visual Studio for Python\watch_photos - Copy\Clock_3\images\val'
output_folder = r'D:\Bai tap\Visual Studio for Python\watch_photos_val_new'
os.makedirs(output_folder, exist_ok=True)

image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

'''total_time = 0
count = 0
clock_count = 0'''

for img_path in image_paths:
    img = cv2.imread(img_path)
    img_base = os.path.basename(img_path)
    output_file = os.path.join(output_folder, img_base)

    '''start = time.time()'''
    results = model.predict(img, verbose=False)
    '''elapsed = time.time() - start

    print(f"{img_path}: inference time = {elapsed:.3f} sec")

    total_time += elapsed
    count += 1'''

    found_clock = False

    for result in results:
        if result.boxes is None:
            continue

        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])

            if cls != CLOCK_CLASS_ID:
                continue  # skip non-clock objects

            found_clock = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]

            base = os.path.splitext(img_base)[0]
            crop_path = os.path.join(output_folder, f"{base}_clock_{i}.jpg")
            cv2.imwrite(crop_path, cropped)

            print("Saved clock crop:", crop_path)

        if found_clock:
            result.save(filename=output_file)
            #print("-> SAVED (Found Clock)")
            #clock_count += 1
        #else:
            #print("-> Skipped")

#if count > 0:
    #print("\n--- Summary ---")
    #print(f"Total images processed: {count}")
    #print(f"Total clocks: {clock_count}")
    #print(f"Total time: {total_time:.2f} sec")
    #print(f"Average inference time: {total_time / count:.3f} sec/image")
    #print(f"FPS: {count / total_time:.2f}")#
