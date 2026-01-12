# START:all
#library for numbers
import numpy as np

#prediction
def predict(X, w, b):   # x reservations, w weight, b bias
    return X * w + b    # predicted pizzas

#measuring errors
def loss(X, Y, w, b):   # y real pizzas
    return np.average((predict(X, w, b) - Y) ** 2)  #mean squared error

#learning to find the best w and b
def train(X, Y, iterations, lr):    # lr learning rate
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        #mini-gradient-descent
        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b #best

    raise Exception("Couldn't converge within %d iterations" % iterations)

#import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

#train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

#predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
# END:all

#plot the chart
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

#real data
sns.set()
plt.plot(X, Y, "bo")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)
x_edge, y_edge = 50, 50
plt.axis([0, x_edge, 0, y_edge])
#learned line
plt.plot([0, x_edge], [b, predict(x_edge, w, b)], linewidth=1.0, color="g")
plt.show()