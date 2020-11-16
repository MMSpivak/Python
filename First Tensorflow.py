import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#t = tf.zeros([5,5,5,5])

#t = tf.reshape(t, [625])
#print(t)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_images[0,23,23]
train_labels[:10]
#print(train_images)
print(train_images[0,23,23])
print(train_labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.figure()                   #plots and shows different images
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#Data preprocessing - this squishes all data between 0 and 1, want data as small as possible so nueral network works faster
train_images = train_images / 255.0

test_images = test_images / 255.0 #if you forget to reprocess test images model wont fit so do both

model = keras.Sequential([ #sequantial is basic, info passes from left to right sequencially
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1), flatten allows us to take in 28 by 28 shape and flatten it into pixels
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2), dense means neurons in previouse layer are connected to all in this one relu -rectify linear unit (lots of different ones)
    keras.layers.Dense(10, activation='softmax') # output layer (3), 10 because only 10 classes, softmax makes sure all values of neurons add to 1 and are between 0 and 1
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2)  # we pass the data, labels and epochs and watch the magic! Less epochs is usually better to not overtune to the training data

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc) #this tests the model on actual data

predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[11])]) #np.argmax shows us the value prediction of the class, class_names will use that # and turn it into the class name
plt.figure()                   #plots and shows different images
plt.imshow(train_images[11])
plt.colorbar()
plt.grid(False)
plt.show()

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number between 1 and 1000 to see the picture: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Error: not a number between 1 and 1000 please try again.")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

