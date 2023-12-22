import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle
df=pd.read_csv('Data/Student_Marks.csv')
x=df['time_study'].values.reshape(-1,1)
y=df['Marks'].values.reshape(-1,1)
lin=LinearRegression()
lin.fit(x,y)
pickle.dump(lin,open('model.pkl','wb'))

