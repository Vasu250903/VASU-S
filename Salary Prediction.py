import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('expected_salary.csv') df.experience = df.experience.fillna('zero')

df

df

df.info()

df.experience = df.experience.astype(int)

df

df.describe()

%matplotlib inline

plt.title('Graph')

plt.xlabel('Experience ')

plt.ylabel('salary')

plt.bar(df.experience, df.salary)

plt.xlabel('Experience ')

plt.ylabel('salary')

plt.plot(df.experience, df.salary)

from sklearn import linear_model

model = linear_model.LinearRegression()

input_data = df.drop(columns='salary')

output_data = df.salary

input_data

output_data

model.fit(input_data,output_data)

model.predict([[2,4,6]])