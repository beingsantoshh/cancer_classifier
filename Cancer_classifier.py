#import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
#input and feature variable
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#output and target name 
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(training_data), len(training_labels))
print(len(validation_data), len(validation_labels))
accuracies = []

for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  #print(classifier.score(validation_data, validation_labels)*100, k)
  accuracies.append(classifier.score(validation_data, validation_labels))

k_list = range(1, 101)  
plt.plot(k_list, accuracies)
plt.show()
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
