# Pirinç sayma
import cv2
import numpy as np

# görüntüyü okuma

img = cv2.imread('rice.jpeg')

# boyutunu değiştirme

frame = cv2.resize(img, (540, 540))

# gri seviyeye dönüştürme

grayscale_Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# eşikleme

th, thresh_img = cv2.threshold(grayscale_Image, 120, 255, cv2.THRESH_BINARY)

# gürültüyü azaltmak için erezyon ve genişlemeyi peşpeşe kullanma. beyaz yerleri incelttiğimiz için beyaz gürültüyü yok ederiz. (erezyon)
# Sonra genişleterek görmek istediğimiz alanı eski haline geri getirmiş oluruz. (genişleme)

kernel = np.ones((5, 5), dtype=np.uint8)
eroded_img = cv2.erode(thresh_img, kernel, iterations=1)
dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

# Gri, binary ve gürültüsü azaltılmış halinin gösterimi

cv2.imshow("gray", grayscale_Image)
cv2.imshow("Dilation", dilated_img)
cv2.imshow("binary", thresh_img)
cv2.waitKey()

# gürültüsü azaltılmış halini etiketleme.
# findContours bize her pirinç tanesinin sınırlarının x,y koordinatlarının arraylerinin listesini döner.(contours)
# cvtColor resmi gri formattan BGR formata çevirir.(COLOR_GRAY2BGR)
# drawContours sınırları belirtilen renk ve kalınlıkta çizer. Bütün sınırları çizmek için -1

contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output_contour = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_contour, contours, -1, (0, 0, 255), 2)

# etiketlenmiş görüntünün gösterimi

cv2.imshow("labeled", output_contour)
cv2.waitKey()

# sınırların koordinatlarının listesinin boyutu bize pirinç sayısını verir.

print("Pirinç tanesi sayısı : ", len(contours))
