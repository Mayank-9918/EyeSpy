    gray_image =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray_image,scale,scaleFactor,minNeighbors)