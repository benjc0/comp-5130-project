import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset
df = pd.read_csv('penguins.csv')

# Visualize raw distribution
plt.figure()
for species, group in df.groupby("species"):
    plt.scatter(group["flipper_length_mm"], group["bill_length_mm"], label=species)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Length (mm)")
plt.legend(title="Species")
plt.title("Raw Penguin Data Distribution")
plt.show()

#Preprocess data
df_clean = df.dropna(subset=['flipper_length_mm', 'bill_length_mm'])




# Feature matrix (X) and label vector (y)
attributes = df_clean[['flipper_length_mm', 'bill_length_mm']].values
type_label = np.where(df_clean['species'] == 'Adelie', 0, 1)

# Create and train the model
model = svm.SVC(kernel='linear')
model.fit(attributes, type_label)


# Extract hyperplane parameters
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(df_clean['flipper_length_mm'].min(),
                 df_clean['flipper_length_mm'].max(), 100)
yy = a * xx - (model.intercept_[0]) / w[1]

# Compute support vector boundary lines
b1 = model.support_vectors_[0]
yy_down = a * xx + (b1[1] - a * b1[0])

b2 = model.support_vectors_[-1]
yy_up = a * xx + (b2[1] - a * b2[0])

# Plot decision boundary
sns.scatterplot(
    x='flipper_length_mm',
    y='bill_length_mm',
    data=df_clean,
    hue='species',
    s=70
)

plt.plot(xx, yy, linewidth=2, color='black', label="Decision Boundary")
plt.plot(xx, yy_down, 'k--', label="Support Vector Boundary")
plt.plot(xx, yy_up, 'k--')

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Length (mm)")
plt.legend()
plt.title("SVM Decision Boundary & Support Vectors")
plt.show()

#Predict and evaluate

def Adelie_or_Gentoo(bill_length_mm, flipper_length_mm):
    """
    Predicts whether a penguin is Adelie or Non-Adelie.
    Inputs must match training format: [flipper_length, bill_length]
    """
    prediction = model.predict([[flipper_length_mm, bill_length_mm]])[0]

    if prediction == 0:
        print(f"FL={flipper_length_mm}, BL={bill_length_mm} → Adelie Penguin")
    else:
        print(f"FL={flipper_length_mm}, BL={bill_length_mm} → Non-Adelie Penguin")


#Demonstration
print("\nPrediction Demonstrations:")
Adelie_or_Gentoo(36.7, 195)
Adelie_or_Gentoo(54.5, 212)