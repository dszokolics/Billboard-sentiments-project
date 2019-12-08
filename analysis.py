import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("lyrics_sentiments2.csv")
dfg = df.groupby('year').agg({'Positive': 'mean', 'Negative': 'mean',
                                             'Mixed': 'mean', 'Neutral': 'mean'})

df_pos_hist = df.Positive.copy()
pos_hist = sns.distplot(df_pos_hist)
pos_hist.set_title('Histogram of Positive Sentiments')
fig3 = pos_hist.get_figure()
fig3.savefig("positive_hist.png")

plt.close()

positive_df = dfg['Positive'].copy()

ax2 = sns.lineplot(data=positive_df)
vals = ax2.get_yticks()
ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax2.set_title('Positive Sentiment over the Years')

fig2 = ax2.get_figure()
fig2.savefig('positive.png')

plt.close()


ax = sns.lineplot(data=dfg)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax.set_title('Sentiment Trends')

fig = ax.get_figure()
fig.savefig('all.png')

plt.close()
