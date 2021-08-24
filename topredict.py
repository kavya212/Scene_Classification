from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model=load_model('Saved_Model_final.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
test_image = image.load_img('predicthockey1.jpeg', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)
print(result)

classof = result.argmax(axis=-1)
print(classof)

if classof== 0:
    print("hockey")
if classof==1:
    print("tennis")


