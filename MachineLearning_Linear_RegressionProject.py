#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# In[89]:


import os
file_path = os.path.join("D:", "Housing.csv")
df = pd.read_csv(file_path)


# In[90]:


df


# In[91]:


df.isnull().sum()


# In[92]:


df.duplicated().sum()


# In[93]:


df.info()


# In[94]:


df.shape


# In[95]:


df.describe(include="all")


# In[96]:


df.hist(figsize=[12,12])
plt.show


# In[97]:


housing_df=pd.get_dummies(df,drop_first=True)
housing_df.head()


# Now we will convert all non numeric data to bool

# In[98]:


boolean_columns = ['mainroad_yes', 'guestroom_yes', 'basement_yes', 
                   'hotwaterheating_yes', 'airconditioning_yes', 
                   'prefarea_yes', 'furnishingstatus_semi-furnished', 
                   'furnishingstatus_unfurnished']

# Convert each column to boolean
housing_df[boolean_columns] = housing_df[boolean_columns].astype(bool)


# In[99]:


housing_df


# In[100]:


df_num=housing_df.select_dtypes(include="number")
df_num


# In[101]:



df_num.head()


# In[102]:


df_num.boxplot()
plt.title("Boxplot with outlier")


# So in above Boxplot we can see we have some outlier

# In[113]:


# Calculate Q1, Q3, and IQR
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1

# Define a condition to filter out outliers
condition = ~((df_num < (Q1 - 1.00 * IQR)) | (df_num > (Q3 + 1.00 * IQR))).any(axis=1)

# Apply the condition to filter the DataFrame
cleaned_data = df_num[condition]
df_cleaned = df.copy()
df_cleaned = df_cleaned.loc[condition.index[condition]]
df_cleaned.reset_index(drop=True, inplace=True)


# So above I have use 1.00 instead of 1.5 because using 1.00 instead of 1.5 makes the threshold for detecting outliers less strict (as with 1.5 there were still lot outliers)

# In[114]:


df_cleaned.boxplot()
plt.title("Boxplot after removing Outliers")


# In[79]:


df_cleaned.columns


# In[117]:


df = df_cleaned.convert_dtypes().select_dtypes("number")
df


# In[141]:


plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='magma', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#Select features based on correlation or domain knowledge
features = df.drop('price', axis=1)
target = df['price']


# Now I will split the data 

# In[120]:



x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)


# In[121]:



y_pred = model_lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 =r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[122]:


n = x_test.shape[0] 
p = x_test.shape[1]  
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f'Adjusted R-squared: {r2_adj}')


# In[140]:


plt.scatter(y_test, y_pred, color='c', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2,color="g")
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

