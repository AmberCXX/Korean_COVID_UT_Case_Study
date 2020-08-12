import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
import warnings 
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns


warnings.filterwarnings("ignore")
time = pd.read_csv("C:/Users/timot/Downloads/datasets_527325_1332417_Time.csv")


# ignoring time, this is what we see.

time["date"] = pd.to_datetime(time["date"].values)

#plt.plot(time["date"],time["test"],'o')
#plt.xlabel("Date")
#plt.ylabel("Tested")
#plt.title("Number of tested over time")
#plt.close()


#plt.plot(time["date"],time["confirmed"])
#plt.xlabel("Date")
#plt.ylabel("Confirmed")
#plt.title("Confirmed cases over time")
#plt.close()

# basic plots for tested vs confirmed
# marginal drop off in more tests
#plt.plot(time["test"],time["confirmed"])
#plt.xlabel("Tested")
#plt.ylabel("Confirmed")

#plt.title("Confirmed vs. Tested")

#plt.plot(time["confirmed"],time["deceased"])
#plt.xlabel("Confirmed")
##plt.ylabel("Deceased")
#plt.title("Confirmed vs. Deceased")


age = pd.read_csv("C:/Users/timot/Downloads/datasets_527325_1332417_TimeAge.csv")

age["date"] = pd.to_datetime(age["date"].values)

old = age[age["age"] == "80s"]
mid = age[age["age"] == "40s"]
young = age[age["age"] == "10s"]

plt.plot(old["date"],old["confirmed"], label = "Old")
plt.plot(mid["date"],mid["confirmed"], label = "Mid")
plt.plot(young["date"],young["confirmed"], label = "Young")

plt.legend()
plt.title("Confirmed cases from 2020-03 to 2020-07")
plt.xticks([])



#plt.plot(old["date"],old["deceased"])
#plt.title("Deceased age 80+ over time")

#plt.plot(mid["date"],mid["deceased"])


#policy = pd.read_csv("C:/Users/timot/Downloads/datasets_527325_1332417_Policy.csv").dropna().drop_duplicates()

#policy["start_date"] = pd.to_datetime(policy["start_date"])
#policy["end_date"] = pd.to_datetime(policy["end_date"])


#TPstart = policy.merge(time,left_on = "start_date",right_on = "date",how = "left").dropna()
#TPend = policy.merge(time,left_on = "end_date",right_on = "date",how = "left").dropna()

#mask = (TPstart["type"] == "Education")
#subset = TPstart[mask]

#plt.plot(subset["date"],subset["confirmed"])


#start = TPstart[["gov_policy","confirmed"]]
#end = TPend[["gov_policy","confirmed"]]

#data = {"Policy":start["gov_policy"], "Start": start["confirmed"],"End": end["confirmed"], "Difference": abs(start["confirmed"] - end["confirmed"])}
#totals = pd.DataFrame(data)

#cases = totals.groupby(by = "Policy").mean()


############################################


#start = TPstart[["type","deceased"]]
#end = TPend[["type","deceased"]]

#data = {"Type":start["type"], "Start": start["deceased"],"End": end["deceased"], "Difference": abs(start["deceased"] - end["deceased"])}
#totals = pd.DataFrame(data)

#cases = totals.groupby(by = "Type").mean()

##############################################

# Tree

patient = pd.read_csv("C:/Users/timot/Downloads/datasets_527325_1332417_PatientInfo.csv",parse_dates = True)
patient["released_date"] = patient["released_date"].fillna(0)
patient["deceased_date"] = patient["deceased_date"].fillna(0)

mask = []

for i in range(len(patient)):
    if patient["released_date"].iloc[i] == 0 and patient["deceased_date"].iloc[i] == 0:
        mask.append(False)
    else:
        mask.append(True)


new = patient[mask]


deceased = []


for i in range(len(new)):
    if new["deceased_date"].iloc[i] != 0:
        deceased.append(1)
    else:
        deceased.append(0)

# 66 dead
# 1651 released
        
new["deceased"] = deceased
        
subset = new[["sex","age","country","province","city","contact_number","deceased"]]

Y,X = dmatrices("deceased~sex + age + country + province + city",subset,return_type = "dataframe")


y = Y.values
model = LogisticRegression()
mse = []

for i in range(0,5):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = i)
    result = model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print(metrics.accuracy_score(y_test,prediction))
    mse.append(metrics.accuracy_score(y_test,prediction))

weights = pd.Series(model.coef_[0],index = X.columns.values)

converted = []
for i in y_test:
    converted.append(i[0])

predicted = list(prediction)
confusion_matrix(converted,predicted)


#fpr, tpr, _ = roc_curve(converted,predicted)

#plt.plot([0,1],[0,1], "k--")
#plt.plot(fpr,tpr)


###############################################################
    
    
model = tree.DecisionTreeClassifier(criterion = "entropy")
mse = []

for i in range(0,10):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = i)
    result = model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print(metrics.accuracy_score(y_test,prediction))
    mse.append(metrics.accuracy_score(y_test,prediction))

converted = []
for i in y_test:
    converted.append(i[0])

predicted = list(prediction)
confusion_matrix(converted,predicted)

graphviz.Source(tree.export_graphviz(model,out_file = None,feature_names= X.columns.values, filled = True))
#os.environ["PATH"] += os.pathsep + "C:/Users/timot/anaconda3/Library/bin/graphviz"
