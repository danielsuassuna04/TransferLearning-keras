# Transfer Learning with ResNet-152 on CIFAR-10

   This project demonstrates how to apply Transfer Learning using the __ResNet-152__ architecture on the __CIFAR-10__ dataset. Transfer learning allows leveraging a pre-trained model (ResNet-152 trained on ImageNet) to improve performance on the target task (CIFAR-10 classification), saving time and computational resources.

## Overview

   The project includes the following key steps:

   1. __Loading the ResNet-152 model__: We use ResNet-50 pre-trained on ImageNet without its final fully connected layers (include_top=False).
   2. __Modifying the ResNet-152 architecture__: We add a global average pooling layer, a dropout layer for regularization, and a final dense layer with softmax activation for classification into the 10 classes of CIFAR-10.

## Steps in the Project

1. ## Loading the Pre-trained ResNet-152 Model:
   We load the ResNet-152 model with weights pre-trained on the ImageNet dataset. The top layers are excluded to allow for customization:
   ```python
   base_model = keras.applications.ResNet152(weights="imagenet", include_top=False)
   ```
2. ## Adding Custom Layers:
   We add several custom layers on top of ResNet-152 for the CIFAR-10 classification task:
   * A global average pooling layer to reduce dimensionality.
   * A dropout layer with a 50% rate to prevent overfitting.
   * A dense layer with the number of classes output units and softmax activation for classification.
3. ## Freezing Layers and Training the Top Layers:
    Initially, we freeze all layers in the ResNet-152 base model to train only the added layers.
   ```python
   for layer in base_model.layers:
      layer.trainable = False
   ```
4. ## Compiling the Model:
  We compile the model with categorical cross-entropy as the loss function and Adam as the optimizer.
   ```python
      optimizer = keras.optimizers.Adam(learning_rate=0.01,weight_decay=0.01)
      model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])
```
5. ## Training the Model:
  We train the model using the CIFAR-10 dataset, first training the custom layers, followed by fine-tuning the entire model.
   ```python
         early = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
   checkpoint = keras.callbacks.ModelCheckpoint(
       filepath='best_model.weights.h5',
       monitor='val_loss',
       save_best_only=True,
       mode='min',
       save_weights_only=True,
       verbose=1
   )
   history = model.fit(train_dataset,epochs=20,validation_data=test_dataset,callbacks=[early,checkpoint])
   ```
6. ## loading the bests weights:
   ```python
   model.load_weights("best_model.weights.h5")
   ```
7. ## Evaluating the Model:
   Finally, we evaluate the model on the test set to measure accuracy and performance.
   ```python
   model.evaluate(test_dataset)
   ```
8. ## Fine-tuning
   After training the custom fully connected layers, you can improve the model's performance by fine-tuning some of the deeper layers in the ResNet-152          
   architecture. Fine-tuning allows the pre-trained layers to adapt to your specific dataset while preserving the knowledge they gained from the ImageNet dataset.

   To fine-tune the model, you need to unfreeze a portion of the ResNet-152 layers and decrease the learning rate. A smaller learning rate ensures that the pre- 
   trained weights are adjusted carefully, preventing drastic updates that could destroy the previously learned features.
9. ## Evaluating the Model again:
   Finally, we evaluate the model again after fine-tuning to check the final accuracy.

   Here’s how to fine-tune the model:
   ```python
   # Unfreeze the last 10 layers of ResNet-152
   for layer in base_model.layers[-10:]:
       layer.trainable = True
   
   # Compile the model with a reduced learning rate
   model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
   ```
      
# Dataset
   The __CIFAR-10 dataset__ consists of 60,000 32x32 color images in 10 different classes, with 50,000 training images and 10,000 test images.
# Results
   * we trained only the new top layers added to the model.
   * The model can be used for CIFAR-10 classification or adapted for other similar tasks using transfer learning.
   * we get 92% of accuracy in the test set
# How to Use
   1. Clone or download this repository.
   2. Install the required dependencies (TensorFlow and Keras).
   3. Run the notebook to train the model.
   4. You can modify the notebook to use your own dataset by replacing the CIFAR-10 data loading code.
# Requirements
   * TensorFlow
   * Keras
   # License
This project is licensed under the MIT License.
   
