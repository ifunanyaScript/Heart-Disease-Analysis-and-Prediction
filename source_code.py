# Import packages needed for analysis and data visualisation 
# %matplotlib inline (Uncomment this line to view the plots)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Pandas dataframe 
df = pd.read_csv(r"c:\Users\USER\Desktop\MyDatasets\heart.csv")
df.head(10)

# Exploring the data  

# Does age affect cholesterol 
sns.set_style('white')
sns.displot(data=df, x='Age', y='Cholesterol', hue='HeartDisease', palette='rocket', cmap='coolwarm')

# Does cholesterol level affect max heart rate
sns.displot(data=df, x='MaxHR', y='Cholesterol')

# Do older people have a higher chance of heart disease?
df1 = df[df.HeartDisease==1]  # Dataframe for heart disease patients only.
plt.figure(figsize=(9, 9))
df1.hist(column='Age',color='#ef283b', grid=False, edgecolor='black')
plt.gca().set_facecolor('#262d99')
plt.title('Age Distribution of Heart Disease')
plt.show()

# How does resting blood pressure affect heart disease?
plt.figure(figsize=(7, 7))
df1.hist(column='RestingBP', color='#d3ef28', grid=False, edgecolor='black')
plt.gca().set_facecolor('#262d99')
plt.show()


# Let's see how the three categorical features varies in heart disease patients
# 1. ChestPainType
a = list(df1.ChestPainType.unique())
b = []
for i in range(len(a)):
    b.append(list(df1.ChestPainType).count(a[i]))
plt.figure(figsize=(7,7))
plt.barh(a, b, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

# 2. RestingECG
c = list(df1.RestingECG.unique())
d = []
for i in range(len(c)):
    d.append(list(df1.RestingECG).count(c[i]))
plt.barh(c, d, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

# 3. ST_Slope
e = list(df1.ST_Slope.unique())
f = []
for i in range(len(e)):
    f.append(list(df1.ST_Slope).count(e[i]))
plt.barh(e, f, color='#cedfd2')
plt.gca().set_facecolor('#262d99')
plt.title('Distribution of chest pain types in heart disease patients')
plt.show()

# We visualise the numerical attributes that might increase the likelihood of heart disease
sns.pairplot(df, hue='HeartDisease', palette='rocket')

# Heat map that shows the correlation between the numerical attributes and heart disease
corr = df.corr()
sns.set(rc = {'figure.figsize':(8, 8)})
sns.heatmap(corr, annot=True)
plt.title('Heat map of correlation between the 6 numerical attributes.')


'''
I recommend you check out the jupyter notebook of this code.
All the analysis and visualisations are therein.
It is in the same repository as this one you're viewing now.
'''

# ifunanyaScript