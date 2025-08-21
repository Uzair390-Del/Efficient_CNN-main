# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.utils import to_categorical
# import numpy as np
# from EnergyEfficientAI import EnergyConsumptionDL

# def load_fashion_mnist():
#     (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#     x_train, x_test = np.expand_dims(x_train, axis=-1) / 255.0, np.expand_dims(x_test, axis=-1) / 255.0
#     y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
#     return x_train, y_train, x_test, y_test

# def experiment_cnn(layers, filter_size, strides, activation, cpu_idle, cpu_full):
#     x_train, y_train, x_test, y_test = load_fashion_mnist()

#     model = Sequential()
    
#     # Add convolutional layers dynamically
#     for i, filters in enumerate(layers):
#         if i == 0:
#             model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, input_shape=(28, 28, 1), padding='same'))
#         else:
#             model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, padding='same'))
        
#         # Apply MaxPooling every 2 layers, but prevent applying it too many times
#         if i % 2 == 1 and i < len(layers) - 1:  # Don't apply pooling too late
#             model.add(MaxPooling2D((2, 2)))

#     # Instead of MaxPooling, use Global Average Pooling to handle smaller dimensions
#     model.add(GlobalAveragePooling2D())  # Global pooling layer to avoid negative dimension issues

#     # Add classification layers
#     model.add(Dense(256, activation=activation))
#     model.add(Dense(10, activation='softmax'))
    
#     # Compile the model
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     # Initialize EnergyConsumptionDL for tracking
#     energy_tracker = EnergyConsumptionDL(model=model, pcpu_idle=cpu_idle, pcpu_full=cpu_full)
    
#     # Generate report with energy consumption details
#     energy_tracker.generate_report(x_train, y_train, x_test, y_test, epochs=10, batch_size=64)

# # # Run experiments for Fashion-MNIST
# # print("\n--- Fashion-MNIST: Baseline Model (8 Layers) ---")
# # experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=10.16, cpu_full=30)

# # print("\n--- Fashion-MNIST: Increased Layers (10 Layers) ---")
# # experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=10.16, cpu_full=30)

# # print("\n--- Fashion-MNIST: Larger Filter Size ---")
# # experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(5, 5), strides=(1, 1), activation='relu', cpu_idle=10.16, cpu_full=30)

# print("\n--- Fashion-MNIST: Sigmoid Activation ---")
# experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=10.16, cpu_full=30)


# # not working
# #  print("\n--- Fashion-MNIST: Larger Strides ---")
# # experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(2, 2), activation='relu', cpu_idle=10.16, cpu_full=30)

# print("\n--- Fashion-MNIST: Increased Layers (10 Layers) --- With Sigmoid function")
# experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=10.16, cpu_full=30)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, axis=-1) / 255.0, np.expand_dims(x_test, axis=-1) / 255.0
    y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test

def experiment_cnn(layers, filter_size, strides, activation, cpu_idle, cpu_full):
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    model = Sequential()
    
    for i, filters in enumerate(layers):
        if i == 0:
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, input_shape=(28, 28, 1), padding='same', kernel_regularizer=l2(0.01)))
        else:
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, padding='same', kernel_regularizer=l2(0.01)))
        
        if i % 2 == 1:
            model.add(Dropout(0.5))  # Add Dropout for regularization

    model.add(GlobalAveragePooling2D())  # Use global pooling
    
    model.add(Dense(256, activation=activation))
    model.add(Dense(10, activation='softmax'))  # Use softmax for multi-class classification
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Fit the model
    model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Run experiments with changes
print("\n--- Fashion-MNIST: Sigmoid Activation ---")
experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=10.16, cpu_full=30)

# print("\n--- Fashion-MNIST: Increased Layers (10 Layers) --- With Sigmoid function")
# experiment_cnn(layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=10.16, cpu_full=30)
