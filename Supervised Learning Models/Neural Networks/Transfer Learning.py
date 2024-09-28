import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load the pre-trained VGG16 model without the top (fully connected) layers
# Weights are pre-trained on ImageNet
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the convolutional base layers so they are not trained
for layer in base_model.layers:
    layer.trainable = False

# Create a new model with the pre-trained base and custom top layers for face recognition
model = models.Sequential([
    base_model,  # Pre-trained base
    layers.Flatten(),  # Flatten the output of the conv base
    layers.Dense(512, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (face or not face)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Data Preprocessing: Assume we have a dataset with face images for training and validation
# ImageDataGenerator for loading and augmenting data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load face dataset from directory
train_data = train_datagen.flow_from_directory(
    'dataset/train',  # Folder with face images
    target_size=(224, 224),  # Resize images to match the input shape of VGG16
    batch_size=32,
    class_mode='binary'  # Binary classification (face vs non-face)
)

validation_data = test_datagen.flow_from_directory(
    'dataset/validation',  # Folder with validation images
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the model on the face dataset
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_data=validation_data,
    validation_steps=validation_data.samples // validation_data.batch_size,
    epochs=10  # You can adjust the number of epochs
)

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# Define data transformations for the training and validation sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the dataset (Assuming it's in a 'dataset' folder with 'train' and 'val' subfolders)
data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for face recognition (assuming binary classification)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (face or not face)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
n_epochs = 10
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}/{n_epochs}')

    # Train phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()
        optimizer.step()

        # Track accuracy and loss
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'face_recognition_resnet18.pth')

# Load the trained model for inference
model.load_state_dict(torch.load('face_recognition_resnet18.pth'))
