import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 数据加载与预处理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化到 [0,1]

# 2. 模型构建（Sequential API）
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 展平输入
    layers.Dense(128, activation='relu'),  # 全连接层
    layers.Dropout(0.2),                   # 防止过拟合
    layers.Dense(10, activation='softmax') # 输出层（10类）
])

# 3. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")