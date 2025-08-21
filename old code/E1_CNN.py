import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from EnergyEfficientAI import EnergyConsumptionDL

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define a function to build and track energy consumption for different CNN models
def experiment_cnn(layers, filter_size, strides, activation, cpu_idle, cpu_full):
    model = Sequential()
    
    # Add convolutional layers dynamically based on input
    for i, filters in enumerate(layers):
        if i == 0:  # First layer specifies input shape
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, input_shape=(28, 28, 1)))
        else:
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation))
        model.add(MaxPooling2D((2, 2)))
    
    # Add classification layers
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Initialize EnergyConsumptionDL for tracking
    energy_tracker = EnergyConsumptionDL(model=model, pcpu_idle=cpu_idle, pcpu_full=cpu_full)
    
    # Generate report with energy consumption details
    energy_tracker.generate_report(x_train, y_train, x_test, y_test, epochs=5, batch_size=64)

# Experiment 1: Baseline model
print("Experiment 1: Baseline")
experiment_cnn(
    layers=[32, 64],
    filter_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100
)

# Experiment 2: Increased layers
print("Experiment 2: Increased Layers")
experiment_cnn(
    layers=[32, 64, 128],
    filter_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100
)

# Experiment 3: Larger filter size
print("Experiment 3: Larger Filter Size")
experiment_cnn(
    layers=[32, 64],
    filter_size=(5, 5),
    strides=(1, 1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100
)

# Experiment 4: Different activation function (Sigmoid)
print("Experiment 4: Sigmoid Activation")
experiment_cnn(
    layers=[32, 64],
    filter_size=(3, 3),
    strides=(1, 1),
    activation='sigmoid',
    cpu_idle=10,
    cpu_full=100
)

# Experiment 5: Larger strides
print("Experiment 5: Larger Strides")
experiment_cnn(
    layers=[32, 64],
    filter_size=(3, 3),
    strides=(2, 2),
    activation='relu',
    cpu_idle=10,
    cpu_full=100
)
