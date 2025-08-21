import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from EnergyEfficientAI import EnergyConsumptionDL

# Function to load dataset (CIFAR-10 or CIFAR-100)
def load_dataset(dataset="cifar10"):
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError("Dataset must be 'cifar10' or 'cifar100'.")

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return x_train, y_train, x_test, y_test, num_classes

# Function to run CNN experiment
def experiment_cnn(dataset, layers, filter_size, strides, activation, cpu_idle, cpu_full):
    x_train, y_train, x_test, y_test, num_classes = load_dataset(dataset)

    model = Sequential()
    
    # Add convolutional layers dynamically
    for i, filters in enumerate(layers):
        if i == 0:  # First layer specifies input shape
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, input_shape=(32, 32, 3), padding='same'))
        else:
            model.add(Conv2D(filters, filter_size, strides=strides, activation=activation, padding='same'))
        
        # Apply MaxPooling every 2 layers instead of after every layer
        if i % 2 == 1:
            model.add(MaxPooling2D((2, 2)))

    # Add classification layers
    model.add(Flatten())
    model.add(Dense(256, activation=activation))  # Increased to 256 neurons
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Initialize EnergyConsumptionDL for tracking
    energy_tracker = EnergyConsumptionDL(model=model, pcpu_idle=cpu_idle, pcpu_full=cpu_full)
    
    # Generate report with energy consumption details
    energy_tracker.generate_report(x_train, y_train, x_test, y_test, epochs=5, batch_size=64)

# Run experiments for CIFAR-10
print("\n--- CIFAR-10: Baseline Model (8 Layers) ---")
experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=10.16, cpu_full=30)

# print("\n--- CIFAR-10: Increased Layers (10 Layers) ---")
# experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=5.39, cpu_full=65)

# print("\n--- CIFAR-10: Larger Filter Size ---")
# experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(5, 5), strides=(1, 1), activation='relu', cpu_idle=5.39, cpu_full=65)

# print("\n--- CIFAR-10: Sigmoid Activation ---")
# experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=5.39, cpu_full=65)

# print("\n--- CIFAR-10: Larger Strides ---")
# experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(2, 2), activation='relu', cpu_idle=5.39, cpu_full=65)

# print("\n--- CIFAR-10: Increased Layers (10 Layers) --- With Sigmoid function")
# experiment_cnn("cifar10", layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=5.39, cpu_full=65)

# Uncomment for CIFAR-100 experiments
# print("\n--- CIFAR-100: Baseline Model (8 Layers) ---")
# experiment_cnn("cifar100", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=10, cpu_full=100)

# print("\n--- CIFAR-100: Increased Layers (10 Layers) ---")
# experiment_cnn("cifar100", layers=[32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024], filter_size=(3, 3), strides=(1, 1), activation='relu', cpu_idle=10, cpu_full=100)

# print("\n--- CIFAR-100: Larger Filter Size ---")
# experiment_cnn("cifar100", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(5, 5), strides=(1, 1), activation='relu', cpu_idle=10, cpu_full=100)

# print("\n--- CIFAR-100: Sigmoid Activation ---")
# experiment_cnn("cifar100", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(1, 1), activation='sigmoid', cpu_idle=10, cpu_full=100)

# print("\n--- CIFAR-100: Larger Strides ---")
# experiment_cnn("cifar100", layers=[32, 64, 128, 128, 256, 256, 512, 512], filter_size=(3, 3), strides=(2, 2), activation='relu', cpu_idle=10, cpu_full=100)
