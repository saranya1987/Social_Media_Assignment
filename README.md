
OBSERVED TREND:
1) BBC has the most negative score compared to all other news.
2) CBS and Fox has equal level of scores compared to other scores
3) Newyork Times has less negative score and stands at the top.



```python
import tweepy
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import consumer_key, consumer_secret, access_token, access_token_secret
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
import time
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser()) 
```


```python
df = pd.DataFrame({"Date":'',"@BBCNews":'',"@CBSNews":'',"@CNN":'',"@FoxNews":'',"@NYT":''}, index=[0])
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@BBCNews</th>
      <th>@CBSNews</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@NYT</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python

target_user = ['@BBCNews', '@CBSNews', '@CNN', '@FoxNews', '@NYT']
```


```python
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

for user in target_user:
    public_tweets = api.user_timeline(user,count=100) 
    counter = 0
# Print Tweets
    for tweet in public_tweets:
        text = tweet['text']
        date = tweet['created_at']
        compound = analyzer.polarity_scores(text)["compound"]
        ts = time.strftime('%m-%d-%Y', time.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
        df['Date'] = ts
        df.set_value(counter, user, compound)
        pos = analyzer.polarity_scores(text)["pos"]
        neu = analyzer.polarity_scores(text)["neu"]
        neg = analyzer.polarity_scores(text)["neg"]
   
        compound_list.append(compound)
        positive_list.append(pos) 
        negative_list.append(neg)
        neutral_list.append(neu)
        counter = counter + 1
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@BBCNews</th>
      <th>@CBSNews</th>
      <th>@CNN</th>
      <th>@FoxNews</th>
      <th>@NYT</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.1027</td>
      <td>0.6103</td>
      <td>0</td>
      <td>0.0258</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.6369</td>
      <td>-0.1779</td>
      <td>-0.2263</td>
      <td>-0.3182</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0258</td>
      <td>-0.6124</td>
      <td>0</td>
      <td>0.1531</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.0772</td>
      <td>-0.2732</td>
      <td>0</td>
      <td>-0.8555</td>
      <td>0.3818</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.34</td>
      <td>0</td>
      <td>-0.2732</td>
      <td>0</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>-0.6597</td>
      <td>0</td>
      <td>-0.4588</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.5719</td>
      <td>-0.5267</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.7506</td>
      <td>-0.886</td>
      <td>-0.7592</td>
      <td>-0.802</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.6249</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.4404</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.5423</td>
      <td>-0.4404</td>
      <td>-0.2732</td>
      <td>0</td>
      <td>0.0258</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0.0772</td>
      <td>-0.2263</td>
      <td>0</td>
      <td>-0.6124</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>-0.7506</td>
      <td>0.015</td>
      <td>-0.5994</td>
      <td>-0.5423</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.296</td>
      <td>-0.6486</td>
      <td>-0.296</td>
      <td>0</td>
      <td>-0.34</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>-0.8807</td>
      <td>0.6926</td>
      <td>-0.34</td>
      <td>0.5719</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0.4754</td>
      <td>-0.4404</td>
      <td>0.6597</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>-0.9246</td>
      <td>0</td>
      <td>0.4019</td>
      <td>0.5106</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.5106</td>
      <td>-0.6597</td>
      <td>0.5413</td>
      <td>0.296</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0.1531</td>
      <td>0.5598</td>
      <td>-0.5267</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0258</td>
      <td>-0.4019</td>
      <td>-0.296</td>
      <td>0</td>
      <td>0.128</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.2732</td>
      <td>0</td>
      <td>0.8625</td>
      <td>0</td>
      <td>-0.128</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.2263</td>
      <td>-0.2732</td>
      <td>0</td>
      <td>0</td>
      <td>-0.34</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.1531</td>
      <td>-0.4939</td>
      <td>0.4404</td>
      <td>0.0772</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.296</td>
      <td>0.1779</td>
      <td>0.1531</td>
      <td>-0.743</td>
      <td>0.2023</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.8402</td>
      <td>0</td>
      <td>0</td>
      <td>-0.296</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>-0.9081</td>
      <td>0.2023</td>
      <td>-0.4854</td>
      <td>0.5267</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>-0.5574</td>
      <td>-0.6486</td>
      <td>-0.3412</td>
      <td>0.4215</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.4417</td>
      <td>-0.872</td>
      <td>-0.2263</td>
      <td>-0.4576</td>
      <td>0.1779</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.3612</td>
      <td>0</td>
      <td>0</td>
      <td>-0.6597</td>
      <td>-0.8316</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.743</td>
      <td>0.4404</td>
      <td>0.5994</td>
      <td>-0.2732</td>
      <td>0.34</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.4404</td>
      <td>-0.5719</td>
      <td>-0.3628</td>
      <td>-0.128</td>
      <td>-0.1531</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.4576</td>
      <td>-0.4939</td>
      <td>0.296</td>
      <td>0.4019</td>
      <td>0.1027</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.1027</td>
      <td>-0.7735</td>
      <td>-0.3818</td>
      <td>-0.4939</td>
      <td>0.4902</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.34</td>
      <td>-0.5267</td>
      <td>0.2023</td>
      <td>-0.7003</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.296</td>
      <td>-0.5994</td>
      <td>-0.4939</td>
      <td>0.0258</td>
      <td>-0.6249</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.2411</td>
      <td>0.4019</td>
      <td>0</td>
      <td>-0.4939</td>
      <td>0.7269</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.5994</td>
      <td>0</td>
      <td>0.4939</td>
      <td>0.3612</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.5719</td>
      <td>0.2481</td>
      <td>0.128</td>
      <td>0</td>
      <td>-0.6249</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.34</td>
      <td>-0.8979</td>
      <td>0</td>
      <td>0</td>
      <td>-0.3818</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.0359</td>
      <td>0.296</td>
      <td>-0.2732</td>
      <td>0.6908</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>79</th>
      <td>-0.4404</td>
      <td>0</td>
      <td>-0.25</td>
      <td>-0.4767</td>
      <td>-0.2878</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0</td>
      <td>-0.7906</td>
      <td>0</td>
      <td>0.6486</td>
      <td>0.0516</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0</td>
      <td>0</td>
      <td>0.128</td>
      <td>-0.7717</td>
      <td>0.7269</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0</td>
      <td>0.5267</td>
      <td>-0.4939</td>
      <td>-0.5267</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>83</th>
      <td>-0.4767</td>
      <td>0</td>
      <td>-0.7003</td>
      <td>-0.7964</td>
      <td>-0.25</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-0.3612</td>
      <td>0</td>
      <td>-0.4939</td>
      <td>-0.4939</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.8718</td>
      <td>-0.4939</td>
      <td>0</td>
      <td>-0.4939</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.3818</td>
      <td>-0.34</td>
      <td>-0.2732</td>
      <td>0.25</td>
      <td>0.34</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>87</th>
      <td>-0.5574</td>
      <td>-0.5574</td>
      <td>0.4847</td>
      <td>-0.4939</td>
      <td>-0.25</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>88</th>
      <td>-0.128</td>
      <td>0.2023</td>
      <td>0</td>
      <td>-0.5994</td>
      <td>-0.7717</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-0.7717</td>
      <td>0.1531</td>
      <td>-0.4939</td>
      <td>-0.4939</td>
      <td>-0.6249</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0</td>
      <td>0.5994</td>
      <td>-0.8074</td>
      <td>0.3182</td>
      <td>-0.8402</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>0.4404</td>
      <td>-0.4939</td>
      <td>0.5267</td>
      <td>-0.4939</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.3818</td>
      <td>-0.34</td>
      <td>0</td>
      <td>-0.6124</td>
      <td>0.5106</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0</td>
      <td>-0.1027</td>
      <td>0.6825</td>
      <td>-0.4939</td>
      <td>-0.2732</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.296</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.0516</td>
      <td>0</td>
      <td>-0.7506</td>
      <td>-0.8176</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>96</th>
      <td>-0.6486</td>
      <td>-0.4939</td>
      <td>-0.2732</td>
      <td>-0.7579</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>97</th>
      <td>-0.8555</td>
      <td>0.1027</td>
      <td>-0.4767</td>
      <td>-0.4939</td>
      <td>-0.5859</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0</td>
      <td>-0.0772</td>
      <td>-0.872</td>
      <td>-0.4939</td>
      <td>-0.743</td>
      <td>03-15-2018</td>
    </tr>
    <tr>
      <th>99</th>
      <td>-0.3818</td>
      <td>0.7351</td>
      <td>0.015</td>
      <td>0</td>
      <td>0</td>
      <td>03-15-2018</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>




```python
df.to_csv("New_mood_data.csv")
mean_c = df.mean()
df2 = pd.DataFrame({'News' : ["BBC","CBS","CNN","Fox","NYT"],'Percent Change' : [mean_c[0],mean_c[1],mean_c[2],mean_c[3],mean_c[4]]})
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>News</th>
      <th>Percent Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC</td>
      <td>-0.229483</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS</td>
      <td>-0.240219</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>-0.093389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox</td>
      <td>-0.192644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NYT</td>
      <td>-0.044879</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("whitegrid", {'grid.linestyle': '--'})
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}                  
sns.set(rc={'figure.figsize':(8,6)})
x = np.arange(100, 0,-1)
plt.ylim(-1,1)
a = plt.scatter(x, df['@BBCNews'], color="skyblue",edgecolors="black",linewidth='1',s=100,label = 'BBC')
b = plt.scatter(x, df['@CBSNews'], color="green",edgecolors="black",linewidth='1',s=100,label = 'CBS')
c = plt.scatter(x, df['@CNN'],color="blue",edgecolors="black",linewidth='1',s=100,label = 'CNN')
d = plt.scatter(x, df['@FoxNews'], color="red",edgecolors="black",linewidth='1',s=100,label = 'Fox')
e = plt.scatter(x, df['@NYT'], color="yellow",edgecolors="black",linewidth='1',s=100,label = 'Newyork Times') 
plt.title('Sentiment Analysis of Media Tweets (03/15/2018)')
plt.xlabel('Number of Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.savefig('Sentiment_tweet.png')
lgnd = plt.legend(bbox_to_anchor=(1, 1),title = 'Media Sources')
plt.gca().invert_xaxis()
plt.show() 
```


![png](output_7_0.png)



```python
# the width of the bars
x=df2['News']
y=df2['Percent Change']
#Condition for setting up the color
colors = ['skyblue' if values == 'BBC' else 'green' if values == 'CBS' else 'yellow' if values == 'NYT' else 'blue' if values == 'CNN' else 'red' for values in x]
rects1 = sns.barplot(x, y, palette=colors)
plt.title('Overall Media Sentiment based on Twitter(3/15/2018)')
plt.ylabel('Tweet Polarity')
plt.savefig('Sentiment_Avg.png')
plt.grid(True)
plt.show()

```


![png](output_8_0.png)

