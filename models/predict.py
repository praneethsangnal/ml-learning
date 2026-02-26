import pandas as pd
import joblib

model=joblib.load("models/logistic_model.pkl")
scaler=joblib.load("models/scaler.pkl")

# new_passenger=pd.DataFrame({
#     "Pclass": [3],
#     "Age":[22],
#     "SibSp":[1],
#     "Parch":[0],
#     "Fare":[7.25],
#     "Sex_male":[True],
#     "Embarked_Q":[False],
#     "Embarked_S":[True]
# })

Pclass=int(input("passenger class (1/2/3)"))
Age=int(input("age"))
SibSp=int(input("total siblings/spouse aboard"))
Parch=int(input("total parents/childern aboard"))
Fare=int(input("Fare"))
Sex_male=input("enter male/female").lower()
Embarked=input("enter C/Q/S").upper()

if(Sex_male=="male"):
    Sex_male=1
else:
    Sex_male=0

Embarked_Q=0
Embarked_S=0
if(Embarked=='S'):
    Embarked_S=1
if(Embarked=='Q'):
    Embarked_Q=1

new_passenger=pd.DataFrame({
    "Pclass":[Pclass],
    "Age":[Age],
    "SibSp":[SibSp],
    "Parch":[Parch],
    "Fare":[Fare],
    "Sex_male":[Sex_male],
    "Embarked_Q":[Embarked_Q],
    "Embarked_S":[Embarked_S]
})



new_passenger=scaler.transform(new_passenger)

predict=model.predict(new_passenger)
probability=model.predict_proba(new_passenger)

print("prediction\n",predict[0])

if(predict[0]==1):
    print("passenger survived\n")
else:
    print("passenger didnt survive\n")

print("survival probability\n",round(probability[0][1]*100,2))