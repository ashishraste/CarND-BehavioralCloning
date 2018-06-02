import model
import data_util as du
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading training and validation data, and creating their respective generators.
samples = du.load_image_file_paths()
training_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = model.image_generator(training_samples, batch_size=32)
validation_generator = model.image_generator(validation_samples, batch_size=32)

model = model.get_nvidia_end_to_end_model()
model.compile(optimizer='adam', loss='mse')
history_obj = model.fit_generator(train_generator, samples_per_epoch=len(training_samples),
                                  validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                  nb_epoch=3, verbose=1)
model.save('model.h5')

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('../images/loss_train_validations.png')


