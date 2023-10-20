from ultralytics import YOLO #основные библиотеки
import cv2
model = YOLO('../Yolo-weights/yolov8n.pt')
# сверху указан "тип размера" котоый нам скачает библиотека ультралайтс, в нашем случае нано(8версия йоло)
results = model(r'C:\Users\HUAWEI\Desktop\MV learning process\Yolo learning\images\1.jpg', show=True)
#Сверху полный путь до изображения
cv2.waitKey(0)
#верхняя строчка нужна чтобы изображение не сворачивалось моментально 
