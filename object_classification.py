import sys
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier


#Load in file and get the first line
in_file = open("INFO521Data.csv",'r')
header_list = in_file.readline().strip().split(",")

#Identify what category we are focusing on
category = "RCompliance"
list_count = ""

if category == "RWeight":
	list_count = 3

if category == "RCompliance":
	list_count = 4

if category == "RWidth":
	list_count = 5
if category == "Object":
    list_count = 6
    
#Get the features into a list along with the response variable
X = []
y = []

for line in in_file:
        line_list = line.strip().split(",")
	y.append(line_list[list_count])
        temp_list = []
	for i in range(7,len(line_list)):
		temp_list.append(float(line_list[i]))
	X.append(temp_list)
	

#Create the model and determine what parameters we want to optimize
#model = svm.SVC(kernel='rbf')
#optimized_parameters = [{'C': [.01, 1, 10, 100, 1000]}]

model = KNeighborsClassifier()
optimized_parameters = [{'n_neighbors': [2,3,4,5,6,7,8,9]}]

#Peform the training
loocv = LeaveOneOut()
clf = GridSearchCV(model, param_grid=optimized_parameters, cv=loocv, n_jobs = -1)
clf.fit(X, y)

#Create an output file
#out_file = open("Results_SVM_" + category + ".txt",'w')
out_file = open(("Results_KNN_" + category + ".txt",'w')

#output desired information
out_file.write("Best Score:\n")
out_file.write(str(clf.best_score_) + "\n\n")
out_file.write("Best parameters set found on development set:\n")
out_file.write(str(clf.best_params_) + "\n\n")
out_file.write("Grid scores on development set:\n")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        out_file.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        out_file.write("\n")

