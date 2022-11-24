import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_rewards(cicles, rewards,loss_path=None,title = "Test"):

    # Load vectors of data
    x = cicles
    y = rewards
    #convert the arrays into list of lists so i can use the in the LinearRegression
    n = np.size(x)
    #print(n)
    x = np.reshape(x,(n,1))
    y = np.reshape(y,(n,1))

    #print(x,y)

    # Linear Regression
    reg = LinearRegression().fit(x, y)
    # Train the model using the training sets
    reg.fit(x, y)
    # Make predictions using the testing set
    y_reg = reg.predict(x)

    # Plot outputs
    plt.figure(1)
    plt.plot(x, y, "b-o", linewidth=0.5)
    plt.plot(x, y_reg, "g--", linewidth=2)
    plt.xlabel("x = Attempts")
    plt.ylabel("y = reward")
    plt.title(title)
    plt.show()

    if loss_path != None:

        loss = np.load("loss_trend.npy")
        # Plot outputs
        plt.figure(2)
        plt.plot(loss, "b-o", linewidth=0.5)
        plt.xlabel("x [attempts]")
        plt.ylabel("y [loss]")
        plt.title("Loss Trend")
        plt.show()
