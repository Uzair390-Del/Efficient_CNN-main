import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from EnergyEfficientAI import EnergyConsumptionDL

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def experiment_cnn(layers, filter_size, strides, activation, cpu_idle, cpu_full, pool_positions):
    model = Sequential()
    
    for i, filters in enumerate(layers):
        # First layer needs input_shape
        if i == 0:
            model.add(Conv2D(filters, filter_size, strides=strides, 
                           padding='same', activation=activation, 
                           input_shape=(28,28,1)))
        else:
            model.add(Conv2D(filters, filter_size, strides=strides,
                           padding='same', activation=activation))
        
        # Only pool at specified layer indices
        if i in pool_positions:
            model.add(MaxPooling2D((2,2)))
    
    # Final classification layers
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    energy_tracker = EnergyConsumptionDL(model=model, pcpu_idle=cpu_idle, pcpu_full=cpu_full)
    energy_tracker.generate_report(x_train, y_train, x_test, y_test, epochs=5, batch_size=64)

# Experiment 1: 8-layer with 3 poolings (positions 1, 3, 5)
print("Experiment 1: 8-layer Baseline")
experiment_cnn(
    layers=[32, 64, 64, 128, 128, 256, 256, 512],  # 8 layers
    filter_size=(3,3),
    strides=(1,1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100,
    pool_positions=[1, 3, 5]  # Pool after layers 1, 3, 5
)

# Experiment 2: 10-layer with 4 poolings
print("\nExperiment 2: 10-layer Model")
experiment_cnn(
    layers=[32, 64, 64, 128, 128, 256, 256, 512, 512, 512],
    filter_size=(3,3),
    strides=(1,1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100,
    pool_positions=[1, 3, 5, 7]  # 4 poolings
)

# Experiment 3: 8-layer with larger filters
print("\nExperiment 3: Larger Filters (5x5)")
experiment_cnn(
    layers=[32, 64, 64, 128, 128, 256, 256, 512],
    filter_size=(5,5),
    strides=(1,1),
    activation='relu',
    cpu_idle=10,
    cpu_full=100,
    pool_positions=[1, 3, 5]
)

# Experiment 4: Different activations
print("\nExperiment 4: Activation Variations")
activations = ['sigmoid', 'tanh', 'elu', 'selu']
for i, act in enumerate(activations, start=4):
    print(f"  {i}. {act} activation")
    experiment_cnn(
        layers=[32, 64, 64, 128, 128, 256, 256, 512],
        filter_size=(3,3),
        strides=(1,1),
        activation=act,
        cpu_idle=10,
        cpu_full=100,
        pool_positions=[1, 3, 5]
    )

# Experiment 5: Larger strides (2,2)
print("\nExperiment 5: Larger Strides (2,2)")
experiment_cnn(
    layers=[32, 64, 64, 128, 128, 256, 256, 512],
    filter_size=(3,3),
    strides=(2,2),  # Strides handle downsampling
    activation='relu',
    cpu_idle=10,
    cpu_full=100,
    pool_positions=[]  # No pooling needed
)