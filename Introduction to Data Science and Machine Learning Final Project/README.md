# Superconductor Discovery via Machine Learning

## Project Overview & Motivation
This project applies a comprehensive suite of Machine Learning and Data Analysis techniques to predict and accelerate the discovery of novel superconductors. By analyzing elemental properties and material compositions, the objective is to accurately predict the critical temperature ($T_c$) of superconductors and computationally synthesize new alloy combinations that exhibit high $T_c$ potential, constrained by realistic physical and chemical parameters.

## Methodology
The analytical framework for this project follows a generalized pipeline applicable to broad materials informatics tasks:

1.  **Data Acquisition and Preprocessing:** * Aggregate numerical feature datasets representing target properties.
    * Standardize the feature space to ensure scale-invariant model performance.
    * Establish robust training and testing splits for unbiased evaluation.
2.  **Baseline Model Selection:** * Evaluate a diverse array of regression algorithms (linear, tree-based, distance-based, and neural networks) to establish a performance baseline.
    * Select the optimal algorithm based on error minimization metrics (e.g., RMSE).
3.  **Hyperparameter Optimization:** * Execute Grid Search with Cross-Validation on the leading model to fine-tune architectural parameters and prevent overfitting.
4.  **Feature Importance and Dimensionality Management:** * Extract built-in feature importance metrics.
    * Apply advanced game-theoretic approaches (SHAP) for robust, interpretable predictor ranking.
    * Iteratively reduce dimensionality to find the optimal balance between computational efficiency and predictive accuracy.
5.  **Anomaly Detection:** * Deploy unsupervised learning techniques (e.g., Support Vector Machines, Isolation Forests) to track high-dimensional outliers.
    * Utilize dimensionality reduction (PCA, t-SNE) to visually map and scientifically cross-check anomalous clusters.
6.  **Surrogate Modeling and Bayesian Optimization:** * Define a binary search space for material composition.
    * Establish domain-specific penalty functions to enforce scientific plausibility (e.g., element count limits, transition metal requirements).
    * Execute Bayesian Optimization to iteratively explore the search space and propose optimal, novel candidates.

## Key Results & Visualizations
* **Predictive Accuracy:** The optimized Random Forest Regressor achieved a Root Mean Squared Error (RMSE) of 8.95.
* **Feature Insights:** SHAP analysis revealed that descriptors related to Thermal Conductivity are the primary drivers for critical temperature prediction. Model efficiency was highly stable using only the top 50 SHAP-ranked features.
* **Novel Synthesis:** The Bayesian Optimization algorithm successfully converged to propose scientifically constrained novel alloys, yielding theoretical candidates such as `HAlArMo` and `HeMnZn` with predicted high critical temperatures.

## Reproducibility & Environment Setup
To ensure a consistent and reproducible environment, this project is packaged for containerized execution. Run the following command in your terminal to spin up the required Jupyter Lab environment and the project's dependencies:


```
pip install pandas numpy matplotlib scikit-learn shap scikit-optimize tqdm cupy
```

```bash
docker run --gpus all -d --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5000:5000 -p 8888:8888 \
  -v /home/erebus/my_tf_projects:/workspace/my_project_container \
  -w /workspace/my_project_container \
  my_custom_tf_env:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=''
```
The dataset can be directly imported using:

```
pip install ucimlrepo
```
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
superconductivty_data = fetch_ucirepo(id=464) 
  
# data (as pandas dataframes) 
X = superconductivty_data.data.features 
y = superconductivty_data.data.targets 
  
# metadata 
print(superconductivty_data.metadata) 
  
# variable information 
print(superconductivty_data.variables) 

```
