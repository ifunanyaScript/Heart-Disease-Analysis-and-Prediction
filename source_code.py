# Import packages needed for analysis and data visualisation 
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Users\USER\Desktop\MyDatasets\heart.csv")
# df.head(10)

sns.set_style('white')
sns.displot(data=df, x='Age', y='Cholesterol', hue='HeartDisease', palette='rocket', cmap='coolwarm')

sns.displot(data=df, x='MaxHR', y='Cholesterol')

df1 = df[df.HeartDisease==1]  # Dataframe for heart disease patients only.
plt.figure(figsize=(9, 9))
df1.hist(column='Age',color='#ef283b', grid=False, edgecolor='black')
plt.gca().set_facecolor('#262d99')
plt.title('Age Distribution of Heart Disease')
plt.show()

plt.figure(figsize=(7, 7))
df1.hist(column='RestingBP', color='#d3ef28', grid=False, edgecolor='black')
plt.gca().set_facecolor('#262d99')
plt.show()

a = list(df1.ChestPainType.unique())
b = []
for i in range(len(a)):
    b.append(list(df1.ChestPainType).count(a[i]))
plt.figure(figsize=(7,7))
plt.barh(a, b, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

c = list(df1.RestingECG.unique())
d = []
for i in range(len(c)):
    d.append(list(df1.RestingECG).count(c[i]))
plt.barh(c, d, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

e = list(df1.ST_Slope.unique())
f = []
for i in range(len(e)):
    f.append(list(df1.ST_Slope).count(e[i]))
plt.barh(e, f, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

sns.pairplot(df, hue='HeartDisease', palette='rocket')

corr = df.corr()
sns.set(rc = {'figure.figsize':(8, 8)})
sns.heatmap(corr, annot=True)
plt.title('Heat map of correlation between the 6 numerical attributes.')
