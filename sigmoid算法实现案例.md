# sigmoid算法实现案例

我决定使用python来实现sigmoid算法，显示出sigmoid算法带来的便携性

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

<!--生成一些示例数据-->

x = np.linspace(-7, 7, 200)
y = sigmoid(x)

<!--绘制 Sigmoid 函数-->

plt.plot(x, y, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()
plt.show()

接下来我通过sigmoid函数来进行简单判断，在一堆考试成绩数据中找出成绩合格的学生和成绩不合格的学生。

import numpy as np
import matplotlib.pyplot as plt

<!--生成一些模拟数据-->

np.random.seed(0)
num_samples = 100
exam_scores = np.random.uniform(30, 100, num_samples)
labels = (exam_scores * 0.5 + np.random.normal(0, 10, num_samples)) > 60

def logistic_regression(x):
    return sigmoid(0.5 * x - 30)

<!--绘制模拟数据和逻辑回归模型-->

plt.scatter(exam_scores, labels, color='blue', label='Training Data')
x_values = np.linspace(30, 100, 100)
plt.plot(x_values, logistic_regression(x_values), color='red', label='Logistic Regression Model')
plt.title('Logistic Regression Model for Admission Prediction')
plt.xlabel('Exam Scores')
plt.ylabel('Admission (1: Admitted, 0: Not Admitted)')
plt.legend()
plt.show()