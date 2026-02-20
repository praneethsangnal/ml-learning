import pandas as pd

df=pd.read_csv("../datasets/titanic.csv")

print("before cleaning")
print(df.isnull().sum())

# I
print("handling missing values- age, cabin, embarked")

print("first age fill with mean")
df["Age"]=df["Age"].fillna(df["Age"].mean())
print(df["Age"])

print("cabin has too many missing values so we drop it")
df=df.drop("Cabin",axis=1)
print(df.head())

print("replace embarked-categorical with mode value")
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])
print("after replacing\n",df["Embarked"].head())

# II
print("dropping useless cols like name,ticke,passid")

df=df.drop(["PassengerId","Name","Ticket"],axis=1)
print(df.head())

# III
print("categorical values need to be encoded now")
df=pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True)



#IV
print("verify\n")
print(df.head())
print(df.isnull().sum())
print(df.shape)
print("type for male",df["Sex_male"].dtype)
