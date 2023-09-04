import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# calcualte co-efficient value
def calc_coef(x, y):
    n = np.size(x)

    # mean value vector of x , y

    mean_x, mean_y = np.mean(x), np.mean(y)

    # cross deviation

    Sxy = np.sum(x * y) - n * mean_x * mean_y
    Sxx = np.sum(x * x) - n * mean_x * mean_x

    # co-efficient

    b1 = Sxy / Sxx
    b0 = mean_y - b1 * mean_x

    return (b0, b1)


# plot regression line
def regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predict_y

    y_predict = b[0] + b[1] * x

    # plot regression line

    plt.plot(x, y_predict, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# main function
def main():
    tips = sns.load_dataset("tips")
    df = pd.DataFrame(tips)
    print(df.head())

    # observation

    x = df["total_bill"]
    y = df["tip"]

    # estimate coefficient function

    b = calc_coef(x, y)

    # plot regression line

    regression_line(x, y, b)


if __name__ == "__main__":
    main()
