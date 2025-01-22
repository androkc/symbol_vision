import numpy as np
import matplotlib.pyplot as plt
import utils


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_pass(image, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output):
    hidden_raw = bias_hidden_input + weights_hidden_input @ image
    hidden = sigmoid(hidden_raw)

    output_raw = bias_hidden_output + weights_hidden_output @ hidden
    output = sigmoid(output_raw)

    return hidden, output


def backpropagation(label, hidden, output, weights_hidden_output):
    delta_output = output - label
    delta_hidden = np.transpose(weights_hidden_output) @ delta_output * sigmoid_derivative(hidden)

    return delta_output, delta_hidden


def update_parameters(weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output,
                      delta_hidden, delta_output, image, hidden, learning_rate):
    weights_hidden_output -= learning_rate * delta_output @ np.transpose(hidden)
    bias_hidden_output -= learning_rate * delta_output

    weights_hidden_input -= learning_rate * delta_hidden @ np.transpose(image)
    bias_hidden_input -= learning_rate * delta_hidden

    return weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output


def train_network(images, labels, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output,
                  epochs, learning_rate):
    for epoch in range(epochs):
        e_loss = 0
        e_correct = 0

        for image, label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))

            hidden, output = forward_pass(image, weights_hidden_input, bias_hidden_input, weights_hidden_output,
                                          bias_hidden_output)

            e_loss += np.mean((output - label) ** 2)
            e_correct += int(np.argmax(output) == np.argmax(label))

            delta_output, delta_hidden = backpropagation(label, hidden, output, weights_hidden_output)

            weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output = update_parameters(
                weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output,
                delta_hidden, delta_output, image, hidden, learning_rate
            )

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {e_loss / len(images):.3f}, Accuracy: {e_correct / len(images) * 100:.2f}%")

    return weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output


def preprocess_image(image_path):
    test_image = plt.imread(image_path, format="jpeg")

    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    test_image = 1 - (gray(test_image).astype("float32") / 255)

    return np.reshape(test_image, (-1, 1))


def predict(image, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output):
    _, output = forward_pass(image, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output)
    return np.argmax(output)


images, labels = utils.load_dataset()

weights_hidden_input = np.random.uniform(-0.5, 0.5, (20, 784))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_hidden_input = np.zeros((20, 1))
bias_hidden_output = np.zeros((10, 1))

epochs = 3
learning_rate = 0.01
weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output = train_network(
    images, labels, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output, epochs,
    learning_rate
)

test_image = preprocess_image("5.jpg")
result = predict(test_image, weights_hidden_input, bias_hidden_input, weights_hidden_output, bias_hidden_output)

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"NN suggests the CUSTOM number is: {result}")
plt.show()
