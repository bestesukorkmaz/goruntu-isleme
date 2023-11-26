#Gerçek Zamanlı nesne tespiti
import requests
import cv2
import numpy as np


url = "http://192.168.1.43:8080/shot.jpg"

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    picture = cv2.imdecode(img_arr, -1)
    # telefon kamerasını kamera olarak kullanır

    frame = cv2.resize(picture, (960, 540))
    # görüntünün boyutunu değiştirir

    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Renk formatını BGR'den HSV'ye dönüştürür.

    L_limit = np.array([0, 100, 100])  # kırmızı alt limiti
    U_limit = np.array([10, 255, 255])  # kırmızı üst limiti

    # kırmızı alt ve üst limitlerini kullanarak sadece kırmızıyı gösterip diğerlerini siyah yapacak şekilde maskeler
    r_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    red = cv2.bitwise_and(frame, frame, mask=r_mask)

    # orijinal görüntü ve maskelenmiş görüntü çıktıları
    cv2.imshow('Original', frame)
    cv2.imshow('red', red)

    # Çıkmak için ESC'ye basın
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
