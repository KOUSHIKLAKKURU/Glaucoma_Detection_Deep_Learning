import sys
import numpy as np
import Preprocessing as pp
import matplotlib.pyplot as plt
import Main_Model as Models
from keras.callbacks import History
from keras.models import Model
from keras.utils import to_categorical


# Plotting Loss and Accuracy Graphs
def plotting(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Training and Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Implement Sequential model to binary classify project data
def project_exp_3():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    ANN_model=Models.DeepANN()
    model = ANN_model.simple_model_bc("sgd")
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=10, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Implement Sequential Model to classify project data by adding various optimization techniques like ADAM, SGD, RMSPROP
def project_exp_4():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    ANN_model = Models.DeepANN()
    model = ANN_model.simple_model_vot("adam") # we can keep sgd, rmsprop also
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=10, validation_data=vv_gen, batch_size=32, callbacks=[history])
    model.save('my_model.h5')
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Implement Sequential Model to classify project data into multiple classes
def project_exp_5():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    ANN_model = Models.DeepANN()
    num_classes=len(np.unique(label))
    model = ANN_model.simple_model_mc("sgd",num_classes)  # we can keep adam, rmsprop also
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=10, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Implement Random-mini-batch evaluations for the above program using python keras
def project_exp_6():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    ANN_model = Models.DeepANN()
    num_classes = len(np.unique(label))
    model = ANN_model.simple_model_mc("adam", num_classes)  # we can keep sgd, rmsprop also
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=10, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Create a convolutional neural network via Keras to classify project data
def project_exp_7():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    CNN_model=Models.DeepCNN()
    num_classes = len(np.unique(label))
    model=CNN_model.cnn_model_aug_reg(num_classes,"rmsprop") # I am not using simple cnn model because it will cause over-fitting
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=15, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Apply augmentation, regularization to the above data and implement cnn by varying cnn architecture
def project_exp_8():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    CNN_model = Models.DeepCNN()
    num_classes = len(np.unique(label))
    model = CNN_model.cnn_model_aug_reg(num_classes)
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=15, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Implement a VGG Model to classify project data by building 3 'blocks' of 2 convolutional layers
def project_exp_9():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    CNN_model = Models.DeepCNN()
    num_classes = len(np.unique(label))
    model=CNN_model.vgg_like_model(num_classes)
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=15, validation_data=vv_gen, batch_size=32, callbacks=[history])
    model.save('my_model.h5')
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Create a Recurrent neural network via Keras to classify  project data
def project_exp_10():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    RNN_model = Models.RNNModel()
    num_classes = len(np.unique(label))
    model = RNN_model.simple_model(num_classes)
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=15, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Generate  LSTM. Model for classification and to identify patterns from the project data set
def project_exp_11():
    data = pp.Preprocessing()
    data.visualize_images('dataset', nimages=5)
    image_df, train, label = data.preprocess1('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    LSTM_model = Models.LSTMModel()
    num_classes = len(np.unique(label))
    model = LSTM_model.simple_model(num_classes)
    print("train generator", tr_gen)
    history = History()
    model.fit(tr_gen, epochs=15, validation_data=vv_gen, batch_size=32, callbacks=[history])
    test_loss, test_acc = model.evaluate(vv_gen)
    print("The ann architecture is")
    print(model.summary())
    print(f"Test Accuracy: {test_acc}")
    plotting(history)

# Use autoencoders to compress the images and implment Resnet to classify data
def project_exp_12():
    data = pp.Preprocessing()
    image_df, train, label = data.preprocess('dataset')
    image_df.to_csv('glaucoma.csv')
    tr_gen, tt_gen, vv_gen = data.generate_train_test_images(image_df)
    num_classes = len(np.unique(label))
    # Create and train the autoencoder
    Auto_Encoder_Model=Models.Auto_Encoders()
    autoencoder = Auto_Encoder_Model.create_autoencoder()
    autoencoder.fit(tr_gen, epochs=50, steps_per_epoch=len(tr_gen), validation_data=vv_gen)

    # Use the encoder to compress the images
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)
    x_train_compressed = encoder.predict(tr_gen)
    x_test_compressed = encoder.predict(tt_gen)

    # Get the class indices from the generators
    y_train = tr_gen.classes
    y_test = tt_gen.classes

    # One-hot encode the labels if necessary
    y_train = to_categorical(y_train, num_classes)
    y_test =to_categorical(y_test, num_classes)
    # Create and train the ResNet model
    resnet = Auto_Encoder_Model.create_resnet(num_classes)
    resnet.fit(x_train_compressed, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    score = resnet.evaluate(x_test_compressed, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    print('''
    1. Implement Sequential model to binary classify project data
    2. Implement Sequential Model to classify project data by adding various optimization techniques like ADAM, SGD, RMSPROP
    3. Implement Sequential Model to classify project data into multiple classes
    4. Implement Random-mini-batch evaluations for the above program using python keras
    5. Create a convolutional neural network via Keras to classify project data
    6. Apply augmentation, regularization to the above data and implement cnn by varying cnn architecture
    7. Implement a VGG Model to classify project data by building 3 'blocks' of 2 convolutional layers
    8. Create a Recurrent neural network via Keras to classify  project data
    9. Generate  LSTM. Model for classification and to identify patterns from the project data set
    10. Use autoencoders to compress the images and implement Resnet to classify data''')
    num=int(input("Which Project Do you want to execute: "))
    if num==1:
        print("Executing project_exp_3: ")
        project_exp_3()
    elif num==2:
        print("Executing project_exp_4: ")
        project_exp_4()
    elif num==3:
        print("Executing project_exp_5: ")
        project_exp_5()
    elif num==4:
        print("Executing project_exp_6: ")
        project_exp_6()
    elif num==5:
        print("Executing project_exp_7: ")
        project_exp_7()
    elif num==6:
        print("Executing project_exp_8: ")
        project_exp_8()
    elif num==7:
        print("Executing project_exp_9: ")
        project_exp_9()
    elif num==8:
        print("Executing project_exp_10: ")
        project_exp_10()
    elif num==9:
        print("Executing project_exp_11: ")
        project_exp_11()
    elif num==10:
        print("Executing project_exp_12: ")
        project_exp_12()
    else:
        print("Invalid experiment number")
        sys.exit()