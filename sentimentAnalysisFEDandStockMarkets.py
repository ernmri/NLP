#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install pandas matplotlib yfinance vaderSentiment


# In[1]:


import pandas as pd

# Load the dataset
fed_data = pd.read_csv('communications.csv', parse_dates=['Release Date'])
print(fed_data.head())


# In[72]:


import yfinance as yf

# Define indices
indices = {
    'DAX': '^GDAXI',  # Germany
    'CAC 40': '^FCHI',  # France
    'FTSE 100': '^FTSE'  # UK
}

# Fetch historical data
stock_data = {}
for name, ticker in indices.items():
    stock_data[name] = yf.download(ticker, start='2023-01-01', end='2023-12-31')

# Preview one of the datasets
print(stock_data['DAX'].head())


# In[73]:


keywords = ['interest rate', 'inflation', 'economic growth']

import re

def extract_relevant_phrases(text, keywords):
    if pd.isnull(text):  # Handle missing text values
        return []
    phrases = []
    for sentence in text.split('.'):  # Split into sentences
        if any(keyword in sentence.lower() for keyword in keywords):  # Check for keywords
            phrases.append(sentence.strip())  # Add relevant sentence
    return phrases


fed_data['Relevant Phrases'] = fed_data['Text'].apply(lambda x: extract_relevant_phrases(x, keywords))

print(fed_data[['Release Date', 'Relevant Phrases']].head())


# In[74]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[75]:


def calculate_sentiment(phrases):
    if not phrases:  # Handle empty lists
        return 0
    total_score = sum(analyzer.polarity_scores(phrase)['compound'] for phrase in phrases)
    return total_score / len(phrases)  # Average score


# In[76]:


fed_data['Sentiment Score'] = fed_data['Relevant Phrases'].apply(calculate_sentiment)


# In[77]:


print(fed_data[['Date', 'Sentiment Score']].head())


# In[78]:


fed_data['Next Trading Day'] = pd.to_datetime(fed_data['Release Date']) + pd.DateOffset(1)


# In[79]:


# Reset the index in both DataFrames to ensure consistency
fed_data = fed_data.reset_index(drop=True)
dax_data = dax_data.reset_index()


# In[80]:


# Ensure 'Adjusted Trading Day' in fed_data and 'Date' in dax_data are datetime
fed_data['Adjusted Trading Day'] = pd.to_datetime(fed_data['Next Trading Day'])
dax_data['Date'] = pd.to_datetime(dax_data['Date'])


# In[81]:


print(fed_data.columns)  # Check column names in fed_data
print(dax_data.columns)  # Check column names in dax_data


# In[82]:


print(fed_data.info())
print(dax_data.info())


# In[83]:


# Flatten multi-level column names in dax_data
dax_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in dax_data.columns]

# Check the new column names
print(dax_data.columns)


# In[84]:


print(fed_data.columns)  # Should include 'Adjusted Trading Day'
print(dax_data.columns)  # Should include 'Date'


# In[85]:


aligned_data = pd.merge(
    fed_data,
    dax_data,
    left_on='Adjusted Trading Day',  # From fed_data
    right_on='Date',                 # From dax_data
    how='inner'                      # Use inner join to keep only matching rows
)


# In[86]:


print(aligned_data.columns)
print(aligned_data.head())


# In[87]:


# Drop redundant columns
aligned_data = aligned_data.drop(columns=['Date_x', 'Date_y'])

# Verify the structure
print(aligned_data.columns)


# In[88]:


# Rename columns
aligned_data.rename(columns={
    'Adjusted Trading Day': 'Trading Day',
    'Close_^GDAXI': 'Close',
    'High_^GDAXI': 'High',
    'Low_^GDAXI': 'Low',
    'Open_^GDAXI': 'Open',
    'Volume_^GDAXI': 'Volume'
}, inplace=True)

# Verify the new column names
print(aligned_data.columns)


# In[89]:


aligned_data['Daily Change (%)'] = aligned_data['Close'].pct_change() * 100


# In[90]:


def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

aligned_data['Sentiment Category'] = aligned_data['Sentiment Score'].apply(categorize_sentiment)


# In[91]:


print(aligned_data.head())


# In[ ]:





# In[92]:


correlation = aligned_data[['Sentiment Score', 'Daily Change (%)']].corr()
print("Correlation Matrix:\n", correlation)


# In[93]:


import matplotlib.pyplot as plt

plt.scatter(aligned_data['Sentiment Score'], aligned_data['Daily Change (%)'])
plt.title('Sentiment Score vs Daily Stock Change')
plt.xlabel('Sentiment Score')
plt.ylabel('Daily Change (%)')
plt.show()


# In[94]:


sentiment_group = aligned_data.groupby('Sentiment Category')['Daily Change (%)'].mean()
sentiment_group.plot(kind='bar', title='Average Stock Change by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Average Daily Change (%)')
plt.show()


# In[95]:


print(aligned_data.isnull().sum())


# In[96]:


aligned_data['Daily Change (%)'].fillna(0, inplace=True)


# In[97]:


# Verify that the required columns exist
required_columns = ['Sentiment Score', 'Daily Change (%)', 'Sentiment Category']

for col in required_columns:
    assert col in aligned_data.columns, f"Column '{col}' is missing in aligned_data!"


# In[98]:


# Check data types of key columns
print(aligned_data[['Sentiment Score', 'Daily Change (%)', 'Sentiment Category']].dtypes)


# In[99]:


aligned_data['Sentiment Category'] = aligned_data['Sentiment Category'].astype('category')


# In[100]:


print(aligned_data[['Sentiment Category']].dtypes)


# In[101]:


print(aligned_data[['Sentiment Score', 'Daily Change (%)']].describe())


# In[102]:


print(aligned_data['Sentiment Category'].value_counts())


# In[103]:


correlation = aligned_data[['Sentiment Score', 'Daily Change (%)']].corr()
print("Correlation Matrix:\n", correlation)


# In[104]:


sentiment_group = aligned_data.groupby('Sentiment Category')['Daily Change (%)'].mean()
print(sentiment_group)


# In[105]:


import matplotlib.pyplot as plt

plt.scatter(aligned_data['Sentiment Score'], aligned_data['Daily Change (%)'])
plt.title('Sentiment Score vs. Daily Stock Change')
plt.xlabel('Sentiment Score')
plt.ylabel('Daily Change (%)')
plt.grid(True)
plt.show()


# In[106]:


sentiment_group.plot(kind='bar', title='Average Stock Change by Sentiment Category')
plt.xlabel('Sentiment Category')
plt.ylabel('Average Daily Change (%)')
plt.show()


# In[107]:


aligned_data['Sentiment Score'].hist(bins=20)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()


# In[108]:





# In[ ]:




