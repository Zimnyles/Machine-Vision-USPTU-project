import cv2
from matplotlib import pyplot as plt 
imaging = cv2.imread(r"C:\Users\HUAWEI\Desktop\proj\venv\vlad.jpg")
#Сверху нужно прописать точный путь до файла
img_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)
imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)

xml_data = cv2.CascadeClassifier(r'C:\Users\HUAWEI\Desktop\proj\venv\haarcascade_frontalface_default.xml')
#Сверху нужно прописать точный путь до файла, в нем я так понял что то типо базы данных объектов
detecting = xml_data.detectMultiScale(img_gray,
                                      minSize = (30,30))
amountDetecting = len(detecting)

if amountDetecting != 0:
    for(a,b,width,height) in detecting:
        cv2.rectangle(imaging_rgb, (a,b),
                      (a + height, b + width),
                      (0,275,0), 9)
        

plt.subplot(1,1,1)
plt.imshow(imaging_rgb)
plt.show(block=True)
