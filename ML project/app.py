import streamlit as st
import pandas as pd
import random
import pickle
import time
from sklearn.preprocessing import StandardScaler
#Title
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.image('https://whatifgaming.com/wp-content/uploads/2023/01/image-1536x864.png.webp')

st.header('A model of housing prices to predict median house values in California using the provided dataset.',divider=True)

# st.subheader('''User must enter given values to predict price:
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House features🏠')

st.sidebar.image('https://wallpaperaccess.com/full/2327343.jpg')

temp_df=pd.read_csv('california.csv')

random.seed(12)
all_values=[]
for i in temp_df[col]:
    min_value,max_value=temp_df[i].agg(['min','max'])

    var=st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
              random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss=StandardScaler()
ss.fit(temp_df[col])

final_value=ss.transform([all_values])

with open('house_priced _pred_ridge_model.pkl','rb') as f:
    gemini=pickle.load(f)

price=gemini.predict(final_value)[0]

st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))

progress_bar=st.progress(0)
placeholder=st.empty()
placeholder.subheader('Predicting Price')
place=st.empty()
place.image('https://media1.tenor.com/m/x3LUiWwPV-MAAAAC/monkey-computer.gif',width=500)

if price > 0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
        
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    
    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)


    
    