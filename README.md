# 🌳 Decision Tree Classifier
**A custom implementation of a Decision Tree Classifier with Gini impurity metric, trained on tabular data, and visualized using NetworkX and Matplotlib.**

**Installation & Setup**

**1. Clone the Repository**
```
git clone https://github.com/your-username/decision-tree-classifier.git
cd decision-tree-classifier
```

**2. Install Required Libraries**
  - Make sure you have Python (preferably 3.8+) installed.
  - Install the required packages:

```
pip install pandas numpy matplotlib networkx openpyxl
```
**Note:**
  - *openpyxl is required for reading/writing Excel files.*
  - *matplotlib is used with the TkAgg backend for better visualization.*

**3. Prepare the Data**
  - Place your training data (trainDATA.xlsx) and test data (testDATA.xlsx) in the project root folder.
  - The last column of trainDATA.xlsx should be the target variable (label).

**4. Run the Project**
```
python decision_tree.py
```
This will:
  - Train a Decision Tree on the training set
  - Predict the labels for the test set
  - Save the predictions into **classification_results.xlsx**
  - Visualize the built decision tree

## 📁 Project Structure
```
decision-tree-classifier/
├── trainDATA.xlsx                # Training dataset
├── testDATA.xlsx                 # Test dataset (features only)
├── decision_tree.py              # Main project file (training, prediction, visualization)
├── classification_results.xlsx   # Output predictions
└── README.md                     # Project description
```
## 🎯 Project Overview
This report presents the implementation of a decision tree classification algorithm to predict the acceptability of cars based on six variables:
1. *Price of the car with 4 categories:* **1 to 4**
2. *Maintenance prices with 4 categories:* **1 to 4**
3. *Number of doors with 4 categories:* **2 to 5**
4. *Capacity of the car with 3 categories:* **2, 4, 6**
5. *Luggage size with 3 categories:* **1 to 3**
6. *Safety with 3 categories:* **1 to 3**
   
It includes:
   - *Data Encoding:* Categorical features are converted to numeric codes.
   - *Gini Index Calculation:* For measuring the impurity at each split.
   - *Tree Building:* Based on minimizing Gini impurity. 
   - *Prediction Function:* Traverses the tree to classify test instances.
   - *Tree Visualization:* Using NetworkX and Matplotlib to display the structure of the decision tree.

## 🧠 Key Functionalities
  - Gini Impurity is used for measuring the "purity" of splits.
  - Recursive Tree Building with depth control (max_depth parameter, default 10).
  - Dynamic Split Finding based on features and threshold values.
  - Leaf Nodes contain the predicted class label.
  - Prediction traverses the tree based on feature values.
  - Visualization shows nodes, edges, and decision paths.

## 📊 Example Workflow
Train the Tree:
   - Load training data
   - Encode categorical columns
   - Build the tree recursively based on Gini impurity
Predict Test Samples:
   - Load test data
   - Traverse the decision tree for each test sample
   - Save predictions in classification_results.xlsx

Visualize:
   - *Nodes:* show feature splits or class predictions
   - *Edges:* show true/false paths based on splits

## 📈 Decision Tree Visualization

**Tree Structure**
  A simplified visualization of the decision tree is shown below:
  ```
[Safety <= 2]
| [Price <= 3]
| | [Luggage Size <= 2] -> Acceptable
| | [Luggage Size > 2] -> Unacceptable
| [Price > 3]
| | -> Unacceptable
[Safety > 2]
| -> Highly Acceptable
```
![image](https://github.com/user-attachments/assets/d713cff7-2224-4bfe-9862-d4433e8c9f1c)
**Figure 1. Result of Decision Tree Visualization**

Visualized with NetworkX and Matplotlib.

**Results:** 
<p>The classification results for the test data are saved in the file classification_results.xlsx.Since the max_depth=5 parameter is a very low value, 
  it causes the tree to fail to generalize and to focus more on the “1” class in the prediction results. But in order not to cause distortion in the decision tree image, 
  the value of the max_depth parameter was used as 5 in the report.</p>
  
![image](https://github.com/user-attachments/assets/d7e40b69-8b0e-4a8b-8a82-13be4084bb3a)

**Figure 2. Result of Predictions with “max_depth=10”**

## Conclusion
<p>This project demonstrates a successful implementation of decision tree classification for car acceptability prediction. 
  The algorithm accurately handles categorical data and provides interpretable results.</p>

## ⚡ Requirements & Notes
- Make sure TkAgg backend is available for Matplotlib if you encounter visualization issues.
- Dataset must be properly formatted:
- *Training Set:* Features + target label (last column)
- *Test Set:* Only features
- Max Depth is adjustable (default max_depth=10) to prevent overfitting.

## 🛠️ Potential Improvements
- Implement pruning to avoid overfitting in deep trees.
- Add entropy support (Information Gain) besides Gini Index.
- Handle missing values and categorical splits better.
- Add Cross-Validation to optimize tree parameters.
