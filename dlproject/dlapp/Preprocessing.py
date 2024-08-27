import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class Preprocessing:
    def visualize_images(self, dir_path, nimages):
        fig, axs = plt.subplots(7, 5, figsize=(10, 10))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            if os.path.isdir(os.path.join(dpath, i)):
                train_class = os.listdir(os.path.join(dpath, i))
                for j in range(nimages):
                    img = os.path.join(dpath, i, train_class[j])
                    img = cv2.imread(img)
                    axs[count][j].title.set_text(i)
                    axs[count][j].imshow(img)
                count += 1
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        # count the number of images in the dataset
        train = []
        labels = []
        for i in os.listdir(dpath):
            # get the list of images in a given class
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                labels.append(i)
        print("number of images:{}\n".format(len(train)))
        print("number of image labels:{}\n".format(len(labels)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': labels})
        print(retina_df)
        return retina_df, train, labels

    def preprocess1(self, dir_path):
        train = []
        labels = []
        for label in os.listdir(dir_path):
            for image_file in os.listdir(os.path.join(dir_path, label)):
                img_path = os.path.join(dir_path, label, image_file)
                train.append(img_path)
                labels.append(label)

        # Convert labels to integers
        le = LabelEncoder()
        labels_int = le.fit_transform(labels)

        # Convert integer labels to one-hot encodings
        labels_one_hot = to_categorical(labels_int)

        print("Number of images:", len(train))
        print("Number of image labels:", len(labels_int))
        df = pd.DataFrame({'Image': train, 'Labels': labels})

        print(df)
        return df, train, labels_one_hot  # Return one-hot encoded labels separately

    def generate_train_test_images(self, df):
        train_df, test_df = train_test_split(df, test_size=0.2)

        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, validation_split=0.15)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=None,  # Set the appropriate directory if necessary
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="binary",
            batch_size=32,
            subset='training'
        )

        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=None,  # Set the appropriate directory if necessary
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="binary",
            batch_size=32,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=None,  # Set the appropriate directory if necessary
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="binary",
            batch_size=32,
            shuffle=False
        )

        print(f"Train images shape: {train_df.shape}")
        print(f"Testing image shape: {test_df.shape}")

        return train_generator, test_generator, validation_generator