from keras.models import Sequential,Model
from keras.layers import Input,UpSampling2D,Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.layers import SimpleRNN
from keras.layers import Reshape, LSTM
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.resnet50 import ResNet50



class DeepANN:
    def simple_model_bc(self, op="sgd"):  # Binary Classification of project data
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if op == "sgd":
            op = SGD(learning_rate=0.001)  # Adjust learning rate here
        model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy'])
        return model

    def simple_model_vot(self, optimizer): # Various Optimization Techniques for multi class classification
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        if optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=0.001)
        elif optimizer.lower() == 'sgd':
            optimizer = SGD(learning_rate=0.001)
        elif optimizer.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=0.001)
        else:
            raise ValueError('Invalid optimizer')
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def simple_model_mc(self,optimizer, num_classes): # Multi-class Classification on project data
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=0.001)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=0.001)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=0.001)
        else:
            raise ValueError('Invalid optimizer')
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                      metrics=['accuracy'])
        return model

class DeepCNN:
    # I am not using because it will become over-fitting
    def simple_cnn_model(self, num_classes, op="sgd"): # Implement CNN Model before applying augmentation and regularization
        cnn = Sequential()
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 3]))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(Flatten())
        cnn.add(Dense(units=128, activation='relu'))
        cnn.add(Dense(units=500, activation="relu"))
        cnn.add(Dense(units=num_classes,
                      activation='softmax'))  # num_classes should be the number of classes in your problem
        cnn.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return cnn

    def cnn_model_aug_reg(self,num_classes, op="sgd", regularizers_strength=0.01): # Implement CNN Model after applying augmentation and regularization
        cnn = Sequential()
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 3]))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(MaxPool2D(pool_size=2, strides=2))
        cnn.add(Flatten())
        cnn.add(Dense(units=128, activation='relu'))
        cnn.add(Dense(500, kernel_regularizer=l2(regularizers_strength), activation="relu"))
        cnn.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(
            regularizers_strength)))  # num_classes should be the number of classes in your problem
        cnn.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return cnn

    def vgg_like_model(self,num_classes, optimizer="sgd"): # Implement VGG Model with 3 'blocks' and 2 CNN Layers.
        model = Sequential()

        # Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Block 2
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # Block 3
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # FC layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

class RNNModel:
    def simple_model(self,n, op="sgd"):  # Add number of classes as a parameter
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))  # Flatten the image data
        model.add(Reshape((-1, 28 * 3)))  # Reshape it into (timesteps, features)
        model.add(SimpleRNN(50, activation='relu', return_sequences=True))
        model.add(SimpleRNN(50, activation='relu'))
        model.add(Dense(n, activation='softmax'))
        model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

class LSTMModel:
    def simple_model(self, n, op="sgd"):  # Add number of classes as a parameter
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 3)))  # Flatten the image data
        model.add(Reshape((-1, 28 * 3)))  # Reshape it into (timesteps, features)
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(n, activation='softmax'))
        model.compile(optimizer=op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

class Auto_Encoders:
    def create_autoencoder(self):
        input_img = Input(shape=(28, 28, 3))  # adapt this if using `channels_first` image data format
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPool2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPool2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    def create_resnet(self,n_classes):
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model