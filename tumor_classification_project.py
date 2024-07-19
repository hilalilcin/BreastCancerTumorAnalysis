import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

########################## ADABOOST CLASSIFIER IMPLEMENTATION ###################
class CustomAdaBoostClassifier:
  
    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2), n_estimators=20):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
                    
    def predict(self, X):
        classifier_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.sign(np.dot(self.alphas, classifier_preds))

    def fit(self, X, y):
        num_samples = X.shape[0]
        self.alphas = np.zeros(self.n_estimators)
        self.trees = np.zeros(self.n_estimators, dtype=object)
        self.sample_weights = np.zeros((self.n_estimators, num_samples))
        self.sample_weights[0] = np.ones(num_samples) / num_samples
        self.errors = np.zeros(self.n_estimators)
        
        for t in range(self.n_estimators):
            sample_weight = self.sample_weights[t]
            tree = self.base_estimator
            tree.fit(X, y, sample_weight=sample_weight)
            tree_pred = tree.predict(X)
            error = sample_weight[(tree_pred != y)].sum()
            alpha = np.log((1 - error) / error) / 2
            updated_weights = sample_weight * np.exp(-alpha * y * tree_pred)
            updated_weights /= updated_weights.sum()

            if t < self.n_estimators - 1:
                self.sample_weights[t + 1] = updated_weights
          
            self.errors[t] = error
            self.trees[t] = tree
            self.alphas[t] = alpha
          
        return self
    
def perform_PCA(data, num_components):
    mean_centered = data - np.mean(data, axis=0)     
    cov_matrix = np.cov(mean_centered, rowvar=False)     
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)     
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_indices]
    sorted_eigenvectors = eigen_vectors[:, sorted_indices]     
    eigenvector_subset = sorted_eigenvectors[:, :num_components]     
    reduced_data = np.dot(eigenvector_subset.T, mean_centered.T).T
     
    return reduced_data

# DATA CLEANING 
data = pd.read_csv("data.csv")
print(data.isna().sum()) # Unnamed: 32 -> this column has 569 NAN value
print("\n")

data = data.drop(columns=['id', 'Unnamed: 32'], axis=1)

data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# FILTER BASED FEATURE SELECTION
plt.figure(figsize=(7, 5), dpi=90)
data.corr(method='pearson')['diagnosis'].sort_values().plot(kind='bar')
data = data.drop(columns=['smoothness_se', 'fractal_dimension_mean', 'texture_se', 'symmetry_se'], axis=1)

plt.figure(figsize=(25, 25))
sns.heatmap(data.corr(), annot=True, fmt='.1f', cmap='Spectral', vmin=-1, vmax=1)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Matrix', size=13)
plt.show()

sns.countplot(data["diagnosis"])
plt.title('Distribution of diagnosis', size=12)

# OUTLIER DETECTION with LOF
data_features = data.drop(['diagnosis'], axis=1)
data_labels = data['diagnosis']

lof = LocalOutlierFactor()
lof_predictions = lof.fit_predict(data_features)
lof_scores = lof.negative_outlier_factor_

outlier_scores_df = pd.DataFrame()
outlier_scores_df['outlier_scores'] = lof_scores

outlier_threshold = -2.5
outliers = outlier_scores_df["outlier_scores"] < outlier_threshold
outlier_indices = outlier_scores_df[outliers].index.tolist()
print("outlier indices = ", outlier_indices)

plt.figure()
plt.scatter(data_features.iloc[outlier_indices, 0], data_features.iloc[outlier_indices, 1], color="black", s=50, label="outlier data")
plt.scatter(data_features.iloc[:, 0], data_features.iloc[:, 1], color="k", s=3, label="normal data")

size = (lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min())
outlier_scores_df["size"] = size
plt.scatter(data_features.iloc[:, 0], data_features.iloc[:, 1], s=1000 * size, edgecolors="g", facecolors="none", label="Outlier Scores")
plt.legend()
plt.show()

data_features = data_features.drop(outlier_indices)
data_labels = data_labels.drop(outlier_indices).values

# FEATURE EXTRACTION USING PCA
scaler = StandardScaler()
data_features = scaler.fit_transform(data_features)
pca_data = perform_PCA(data_features, 2)

pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2'])

# CLASSIFICATION
X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.25, random_state=42)

models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=60, random_state=0),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=20),
    'CustomAdaBoostClassifier': CustomAdaBoostClassifier()
}

print("\nClassification accuracy results using different techniques\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: {accuracy * 100:.2f}%")
