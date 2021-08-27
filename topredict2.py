from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv
model=load_model('Saved_Model_final.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
cap = cv.VideoCapture('predicttennisvid2.mp4')
while(1):
    # Take each frame
    _, frame = cap.read()
    #cv.imshow('frame',frame)
    test_image = image.img_to_array(frame)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image) 
    classof = result.argmax(axis=-1)
    if classof== 0:
        text="hockey"
    if classof==1:
        text="tennis"
    res = cv.putText(frame,text,(50,50),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv.LINE_AA)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()


#predict the result
'''result = model.predict(test_image)
print(result)


print(classof)'''


