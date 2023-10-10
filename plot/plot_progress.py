import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("progress.txt", delimiter="\t")


def plot_and_save(column):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"].values, df[column].values)
    plt.scatter(df["Epoch"].values, df[column].values)
    plt.xlabel("Epoch")
    plt.ylabel(column)
    plt.title(column)
    plt.grid(True)
    plt.savefig(column + ".png")
    # plt.show()


# Plot AverageEpRet
plot_and_save("AverageEpRet")

# Plot AverageEpCost
plot_and_save("AverageEpCost")

# Plot AverageTestEpCost
plot_and_save("AverageTestEpCost")

# Plot LossAlpha
plot_and_save("LossAlpha")
