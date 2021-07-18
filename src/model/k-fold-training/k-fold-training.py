import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import optimizers

from utils.split_data_in_k_folds import create_kfold_directories
from utils.split_data_in_k_folds import populate_kfold_directories


DATA_DIR = '../data/train_data'
TEST_DIR = '../data/test_data'
IMAGE_SIZE = (150, 150)


""" HYPER PARAMETERS """
EPOCHS = 25
STEPS_PER_EPOCH = 10
K_FOLDS = 5

# batch size = number of items in train_dir / STEPS_PER_EPOCH
TRAIN_BATCH_SIZE = int((400 - (400/K_FOLDS)) / STEPS_PER_EPOCH)
# batch size = number of items in val_dir / STEPS_PER_EPOCH
VALIDATION_BATCH_SIZE = int((400 / K_FOLDS) / STEPS_PER_EPOCH)


def create_model():
    model = models.Sequential()
    model.add(VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    # Only train last convolutional layers of the convolutional base
    for layer in model.layers[0].layers:
        if layer.name == 'block5_conv1':
            break
        else:
            layer.trainable = False

    return model


def create_data_generators():
    # augmentation in training data
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    return train_datagen, val_datagen


def create_train_generator(train_datagen, train_dir, batch_size=TRAIN_BATCH_SIZE):
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                        class_mode='categorical')
    return train_generator


def create_val_generator(val_datagen, val_dir, batch_size=VALIDATION_BATCH_SIZE):
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=batch_size,
                                                           class_mode='categorical')
    return validation_generator


def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])


def fit_model(model, train_generator, validation_generator):
    history = model.fit(train_generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                        validation_data=validation_generator, validation_steps=STEPS_PER_EPOCH)
    return history


def plot_results(history, i):

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(range(EPOCHS), accuracy, 'bo', label='Training acc')
    plt.plot(range(EPOCHS), val_accuracy, 'ro', label='Validation acc')
    plt.title('Training and Validation accuracy fold {}'.format(i+1))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure()

    plt.plot(range(EPOCHS), loss, 'bo', label='Training loss')
    plt.plot(range(EPOCHS), val_loss, 'ro', label='Validation loss')
    plt.title('Training and Validation loss fold {}'.format(i+1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_avg(acc, val_acc, loss, val_loss):

    plt.plot(range(EPOCHS), acc, 'bo', label='Training accuracy')
    plt.plot(range(EPOCHS), val_acc, 'ro', label='Validation accuracy')
    plt.title(f'{K_FOLDS}-fold Training and Validation average accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure()

    plt.plot(range(EPOCHS), loss, 'bo', label='Training loss')
    plt.plot(range(EPOCHS), val_loss, 'ro', label='Validation loss')
    plt.title(f'{K_FOLDS}-fold Training and Validation average loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def train():
    # create k-fold sets
    create_kfold_directories(K_FOLDS)
    populate_kfold_directories(DATA_DIR, K_FOLDS)

    k_fold_histories = []
    for i in range(K_FOLDS):
        # create new model for every fold
        model = create_model()
        compile_model(model)

        history = train_one_fold(model, i)
        k_fold_histories.append(history)
        plot_results(history, i)

    process_histories(k_fold_histories)


def train_one_fold(model, i):

    # Create needed resources
    train_dir = f'folds/fold{i}/train'
    val_dir = f'folds/fold{i}/validation'
    train_datagen, val_datagen = create_data_generators()

    train_generator = create_train_generator(train_datagen, train_dir)
    print(train_generator.class_indices)
    val_generator = create_val_generator(val_datagen, val_dir)
    print(val_generator.class_indices)

    return fit_model(model, train_generator, val_generator)


def process_histories(k_fold_histories):

    accuracies = [k_fold_histories[i].history['accuracy'] for i in range(K_FOLDS)]
    losses = [k_fold_histories[i].history['loss'] for i in range(K_FOLDS)]
    val_accuracies = [k_fold_histories[i].history['val_accuracy'] for i in range(K_FOLDS)]
    val_losses = [k_fold_histories[i].history['val_loss'] for i in range(K_FOLDS)]

    # find average values
    accuracy = sum(map(np.array, accuracies)) / K_FOLDS
    loss = sum(map(np.array, losses)) / K_FOLDS
    val_accuracy = sum(map(np.array, val_accuracies)) / K_FOLDS
    val_loss = sum(map(np.array, val_losses)) / K_FOLDS

    plot_avg(accuracy, val_accuracy, loss, val_loss)


def train_final_model():

    # create the model
    model = create_model()
    compile_model(model)

    # prepare data
    train_data_gen, _ = create_data_generators()
    train_generator = create_train_generator(train_data_gen, DATA_DIR, 40)

    # train and save the model
    model.fit(train_generator, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)
    model.save('model.h5')
    model.save_weights('model_weights.h5')

    # evaluate the model
    evaluate_with_test_data(model)


def evaluate_with_test_data(model):

    labels_dic = {"Alarmed": 0, "Annoyed": 1, "Curious": 2, "Relaxed": 3}
    test_images = image_dataset_from_directory(TEST_DIR, image_size=IMAGE_SIZE, labels="inferred", label_mode='categorical')

    labels = [labels_dic[image.split("/")[3]] for image in test_images.file_paths]
    predictions = [np.argmax(prediction) for prediction in model.predict(test_images)]

    hits = [labels[i] == predictions[i] for i in range(len(labels))].count(False)
    print(f"Test set accuracy {(hits/len(labels))*100}%")


if __name__ == "__main__":
    train()
    train_final_model()
