#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium 
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go

plt.rcParams['figure.figsize'] = 12,14


# In[2]:


covid_cases = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/Covid cases in India.xlsx')


# In[3]:


covid_cases.head()


# In[4]:


covid_cases.drop(['S. No.'],axis=1,inplace=True)


# In[5]:


covid_cases['Total Cases'] = covid_cases['Total Confirmed cases (Indian National)'] + covid_cases['Total Confirmed cases ( Foreign National )']


# In[6]:


total_cases = covid_cases['Total Cases'].sum()
print('Total Number of cases in India till 22nd March 2020: ',total_cases)


# In[7]:


covid_cases.style.background_gradient(cmap = 'YlGn')


# In[8]:


covid_cases['Total Active'] = covid_cases['Total Cases'] - (covid_cases['Cured'] + covid_cases['Death'])


# In[9]:


total_cases = covid_cases['Total Active'].sum()


# In[10]:


print('Total number of active cases in India till 22nd March 2020: ',total_cases)


# In[11]:


Total_act_cases = covid_cases.groupby('Name of State / UT')['Total Active'].sum().sort_values(ascending = False).to_frame()


# In[12]:


Total_act_cases.style.background_gradient(cmap = 'Blues')


# In[13]:


Indian_Coordinates = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/Indian Coordinates.xlsx')


# In[14]:


Indian_Coordinates.head()


# In[15]:


merged_data = pd.merge(Indian_Coordinates,covid_cases,on = 'Name of State / UT')


# In[16]:


map = folium.Map(location = [20,70],zoom_start = 4,tiles ='Stamenterrain')
map


# In[17]:


for lat,lon,value,name in zip(merged_data['Latitude'],merged_data['Longitude'],
                              merged_data['Total Cases'],merged_data['Name of State / UT']):
    folium.CircleMarker([lat,lon],radius=value*0.7,color='#007849',fill_color='green',
                        popup = ('<strong>State</strong>: '+ str(name).capitalize())).add_to(map)
map            
    


# In[18]:


f,ax = plt.subplots(figsize=(14,10))
cured_vs_total = merged_data[['Name of State / UT','Total Cases','Cured','Death']]
cured_vs_total.sort_values('Total Cases',ascending=True,inplace=True)


sns.barplot(x ='Total Cases',y ='Name of State / UT', data = cured_vs_total,label='Total',color = '#CCE5FF')
sns.barplot(x = 'Cured',y='Name of State / UT', data = cured_vs_total,label='Cured',color = '#B2FF66')
ax.legend(ncol=2,loc = 'center right',frameon = True)
ax.set(xlabel='Cases',ylabel='Name of State / UT')
sns.despine(left = True,bottom = True)


# In[19]:


f,ax = plt.subplots(figsize=(14,10))
deaths_vs_total = merged_data[['Name of State / UT','Total Cases','Cured','Death']]
deaths_vs_total.sort_values('Total Cases',ascending=True,inplace=True)


sns.barplot(x ='Total Cases',y ='Name of State / UT', data = cured_vs_total,label='Total',color='#CCE5FF')
sns.barplot(x = 'Death',y='Name of State / UT', data = cured_vs_total,label='Deaths',color = '#FF6666')
ax.legend(ncol=2,loc = 'center right',frameon = True)
ax.set(xlabel='Cases',ylabel='Name of State / UT')
sns.despine(left = True,bottom = True)


# In[20]:


per_day_cases = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/per_day_cases.xlsx')


# In[21]:


per_day_cases.head()


# In[62]:


Scatter = px.scatter(per_day_cases,x = per_day_cases['Date'],y = per_day_cases['Total Cases'])
Scatter.update_layout(title_text = 'Trends of Coronavirus in India ',plot_bgcolor = '#E9ECEF',font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x=0.5)
Scatter.show()


# In[23]:


bar = px.bar(per_day_cases,x = per_day_cases['Date'],y = per_day_cases['New Cases'])
bar.update_layout(title_text = 'Trends of Coronavirus in India Day Wise ',plot_bgcolor = '#E9ECEF',font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
bar.show()


# In[24]:


perday_India = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/PerDayCases(India).xlsx')


# In[25]:


perday_India.head()


# In[26]:


perday_Italy = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/PerDayCases(Italy).xlsx')


# In[27]:


perday_Italy.head()


# In[28]:


perday_Korea = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/PerDayCases(Korea).xlsx')


# In[29]:


perday_Korea.head()


# In[30]:


perday_Wuhan = pd.read_excel('C:/Users/DELL/Desktop/Minor Project/PerDayCases(Wuhan).xlsx')


# In[31]:


perday_Wuhan.head()


# In[32]:


bar = px.bar(perday_India,x = 'Date',y = 'Total Cases')
bar.update_layout(title_text = 'Situation in India ',plot_bgcolor = '#E9ECEF',
                  font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
bar.show()


# In[33]:


bar = px.bar(perday_Italy,x = 'Date',y = 'Total Cases')
bar.update_layout(title_text = 'Situation in Italy ',plot_bgcolor = '#E9ECEF',
                  font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
bar.show()


# In[34]:


bar = px.bar(perday_Korea,x = 'Date',y = 'Total Cases')
bar.update_layout(title_text = 'Situation in Korea ',plot_bgcolor = '#E9ECEF',
                  font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
bar.show()


# In[35]:


bar = px.bar(perday_Wuhan,x = 'Date',y = 'Total Cases')
bar.update_layout(title_text = 'Situation in Wuhan ',plot_bgcolor = '#E9ECEF',
                  font_color = 'black',font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
bar.show()


# In[36]:


after100_India = px.scatter(perday_India,x='Days after surpassing 100 cases',y='Total Cases')
after100_India.update_layout(title_text = 'Situation in India after surpassing 100 cases ',
                             plot_bgcolor = '#E9ECEF',font_color = 'black',
                             font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
after100_India.show()


# In[37]:


after100_Italy = px.scatter(perday_Italy,x='Days after surpassing 100 cases',y='Total Cases')
after100_Italy.update_layout(title_text = 'Situation in Italy after surpassing 100 cases ',
                             plot_bgcolor = '#E9ECEF',font_color = 'black',
                             font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
after100_Italy.show()


# In[38]:


after100_Korea = px.scatter(perday_Korea,x='Days after surpassing 100 cases',y='Total Cases')
after100_Korea.update_layout(title_text = 'Situation in Korea after surpassing 100 cases ',
                             plot_bgcolor = '#E9ECEF',font_color = 'black',
                             font_family = 'Times New Roman',font_size = 14,title_x = 0.5)
after100_Korea.show()


# In[39]:


worldwide_data = pd.read_csv('C:/Users/DELL/Desktop/Minor Project/covid_19_clean_complete.csv',parse_dates = ['Date'])


# In[40]:


worldwide_data.head()


# In[41]:


worldwide_data.rename(columns = {'Country/Region':'Country'},inplace=True)


# In[42]:


worldwide_data.drop(['Lat','Long'],axis=1,inplace=True)


# In[43]:


worldwide_data.head()


# In[44]:


world_frame = worldwide_data.groupby('Date',as_index = False).sum()


# In[45]:


world_frame.head(10)


# In[46]:


world_frame.tail(10)


# In[47]:


from fbprophet import Prophet


# In[48]:


confirmed = world_frame.groupby('Date').sum()['Confirmed'].reset_index()
deaths = world_frame.groupby('Date').sum()['Deaths'].reset_index()
recovered = world_frame.groupby('Date').sum()['Recovered'].reset_index()


# In[49]:


confirmed.head()


# In[50]:


confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])


# In[51]:


model = Prophet(interval_width = 0.90)
model.fit(confirmed)
future = model.make_future_dataframe(periods = 12)
future.tail()


# In[52]:


forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[53]:


plot_forecast = model.plot(forecast)


# In[54]:


deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])


# In[55]:


model = Prophet(interval_width = 0.90)
model.fit(deaths)
future = model.make_future_dataframe(periods = 12)
future.tail()


# In[56]:


forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[57]:


plot_forecast = model.plot(forecast)


# In[58]:


recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])


# In[59]:


model = Prophet(interval_width = 0.90)
model.fit(recovered)
future = model.make_future_dataframe(periods = 12)
future.tail()


# In[60]:


forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[61]:


plot_forecast = model.plot(forecast)


# In[ ]:




