import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

# Load the CSV file
df = pd.read_csv("penguins.csv")

# Create scatter plot: flipper_length_mm (x) vs bill_length_mm (y)
plt.figure()
for species, group in df.groupby("species"):
    plt.scatter(
        group["flipper_length_mm"],
        group["bill_length_mm"],
        label=species
    )

# Labels and legend
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Length (mm)")
plt.legend(title="Species")

# Show plot
plt.show()

