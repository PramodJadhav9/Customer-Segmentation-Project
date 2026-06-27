import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create sample customer data
np.random.seed(42)

customers = 100

data = {
    "CustomerID": range(1, customers + 1),
    "Age": np.random.randint(18, 60, customers),
    "AnnualIncome": np.random.randint(20000, 120000, customers),
    "SpendingScore": np.random.randint(1, 100, customers)
}

df = pd.DataFrame(data)

# Save the dataset
df.to_csv("customer_data.csv", index=False)

# Select features for clustering
X = df[["AnnualIncome", "SpendingScore"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X)

# Print first 5 customers
print(df.head())

# Draw the graph
for segment in df["Segment"].unique():
    temp = df[df["Segment"] == segment]
    plt.scatter(
        temp["AnnualIncome"],
        temp["SpendingScore"],
        label=f"Segment {segment}"
    )

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.legend()

# Save the graph
plt.savefig("customer_segments.png")

# Show the graph
plt.show()