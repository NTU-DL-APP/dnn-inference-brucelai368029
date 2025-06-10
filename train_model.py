import tensorflow as tf

# 下載並載入 Fashion-MNIST 資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize 0~1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 建立簡單模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# 測試準確率
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# 儲存模型
model.save("fashion_mnist.h5")
