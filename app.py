import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sklearn as sk    

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#os.chdir("C:\\Users\\chris\\OneDrive\\Chris\\MSBA\\Programming_II\\Data_Sets")


# In[4]:


s = pd.read_csv("social_media_usage.csv")






# In[7]:


def clean_sm(x):
    x = np.where(x == 1,
                 1,
                 0)
    return(x)


# In[14]:


df = pd.DataFrame({"a": [1,2,3], "b": [1,1,3]})


# In[16]:


clean_sm(df)





# In[17]:


ss = pd.DataFrame({
    "sm_li":clean_sm(s.web1h),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] ==2, 1, 0),
    "age":np.where(s["age"] > 97, np.nan, s["age"]),
})


# In[18]:


ss = ss.dropna()



# In[23]:


y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]



# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state= 456)




# In[25]:


lr = LogisticRegression(class_weight = "balanced")


# In[26]:


lr.fit(X_train, y_train)





# In[27]:

y_pred = lr.predict(X_test)

# In[ ]:
# 
# 
st.title('LinkedIn User Prediction')

st.markdown('1. Income')

st.write(pd.DataFrame({
    "Level": [1,2,3,4,5,6,7,8,9],
    "Income": ["Less than $10,000", "$10,000 to under $20,000", "$20,000 to under $30,000", "$30,000 to under $40,000", "$40,000 to under $50,000", "$50,000 to under $75,000",
    "$75,000 to under $100,000", "$100,000 to under $150,000", "$150,000 or more"]}))

#commenting out the slider
#i = st.slider('Income', 1, 9, 5)

i = st.selectbox(
    'Please reference the above table and select your income levle from the dropdownm',
   (1,2,3,4,5,6,7,8,9), key= int)


st.markdown("2. Education")

st.write(pd.DataFrame({
    "Level": [1,2,3,4,5,6,7,8],
    "Education": ["Less than highschool (grades 1-8 or no formal schooling", "Highschool incomplete", "Highschool graduate", "Some college, no degree",
    "Two-year associate degree from college or university", "Four-year college or university degree", "Some postgraduate or professional schooling, no degree",
    "Postgraduate or professional degree"]}))

e = st.selectbox(
    'Please reference the above table and select your education level from the dropdownm',
   (1,2,3,4,5,6,7,8))

#commenting out the slider
#e = st.slider('Please reference the aboce table and select your education Level using the slider', 1, 8, 3)

st.markdown("4. Are you a parent? Yes = 1, No = 0")
p = st.slider('Parent', 1, 0)

st.markdown("5. Are you married? Yes = 1, No = 0")
m = st.slider('Married', 1, 0)

st.markdown("6. What is your gender? Female = 1, Other = 0")
g = st.slider('Gender', 1, 0)

st.markdown("7. Please select your age")
a = st.slider('Age', 1, 97, 35)

persona = [i, e, p, m, g, a]

probability = lr.predict_proba([persona])
prob_r = probability[0][1]

st.markdown("This logistic regression model predicts that a person with the above attributes:")
if prob_r >= 0.5:
    st.write("is a LinkedIn user.")
else:
        st.write("is not a LinkedIn user.")


st.markdown("The probability that this person is a LinkedIn user is:")
st.write(round((prob_r),2))



# %%
