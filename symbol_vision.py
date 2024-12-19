import numpy as np
import matplotlib.pyplot as plt

import utils

images, labels = utils.load_dataset()

# Матрица весов
weights_hidden_input = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))

# Нейронные смещения
bias_hidden_input = np.zeros((20, 1))
bias_hidden_output = np.zeros((10, 1))

epochs = 3
e_loss = 0
e_correct = 0
learning_rate = 0.01

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        hidden_raw =  bias_hidden_input + weights_hidden_input @ image
        hidden = 1 / (1 + np.exp(-hidden_raw))

        output_raw = bias_hidden_output + weights_hidden_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        e_loss += 1 / len(output_raw) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        delta_output = output - label
        weights_hidden_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_output += -learning_rate * delta_output

        # Backpropagation (hidden layer)
        delta_hidden = np.transpose(weights_hidden_output) @ delta_output * (hidden * (1 - hidden))
        weights_hidden_input += -learning_rate * delta_hidden @ np.transpose(image)
        bias_hidden_input += -learning_rate * delta_hidden

    print(f"Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

# CHECK CUSTOM
test_image = plt.imread("5.jpg", format="jpeg")

# Grayscale + Unit RGB + inverse colors
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
test_image = 1 - (gray(test_image).astype("float32") / 255)

# Reshape
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

# Predict
image = np.reshape(test_image, (-1, 1))

# Forward propagation (to hidden layer)
hidden_raw = bias_hidden_input + weights_hidden_input @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid
# Forward propagation (to output layer)
output_raw = bias_hidden_output + weights_hidden_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the CUSTOM number is: {output.argmax()}")
plt.show()