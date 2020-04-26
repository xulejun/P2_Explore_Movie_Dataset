
# coding: utf-8

# ## 探索电影数据集
# 
# 在这个项目中，你将尝试使用所学的知识，使用 `NumPy`、`Pandas`、`matplotlib`、`seaborn` 库中的函数，来对电影数据集进行探索。
# 
# 下载数据集：
# [TMDb电影数据](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/explore+dataset/tmdb-movies.csv)
# 

# 
# 数据集各列名称的含义：
# <table>
# <thead><tr><th>列名称</th><th>id</th><th>imdb_id</th><th>popularity</th><th>budget</th><th>revenue</th><th>original_title</th><th>cast</th><th>homepage</th><th>director</th><th>tagline</th><th>keywords</th><th>overview</th><th>runtime</th><th>genres</th><th>production_companies</th><th>release_date</th><th>vote_count</th><th>vote_average</th><th>release_year</th><th>budget_adj</th><th>revenue_adj</th></tr></thead><tbody>
#  <tr><td>含义</td><td>编号</td><td>IMDB 编号</td><td>知名度</td><td>预算</td><td>票房</td><td>名称</td><td>主演</td><td>网站</td><td>导演</td><td>宣传词</td><td>关键词</td><td>简介</td><td>时常</td><td>类别</td><td>发行公司</td><td>发行日期</td><td>投票总数</td><td>投票均值</td><td>发行年份</td><td>预算（调整后）</td><td>票房（调整后）</td></tr>
# </tbody></table>
# 

# **请注意，你需要提交该报告导出的 `.html`、`.ipynb` 以及 `.py` 文件。**

# 
# 
# ---
# 
# ---
# 
# ## 第一节 数据的导入与处理
# 
# 在这一部分，你需要编写代码，使用 Pandas 读取数据，并进行预处理。

# 
# **任务1.1：** 导入库以及数据
# 
# 1. 载入需要的库 `NumPy`、`Pandas`、`matplotlib`、`seaborn`。
# 2. 利用 `Pandas` 库，读取 `tmdb-movies.csv` 中的数据，保存为 `movie_data`。
# 
# 提示：记得使用 notebook 中的魔法指令 `%matplotlib inline`，否则会导致你接下来无法打印出图像。

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 不要忘记 pyplot
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
movie_data = pd.read_csv('tmdb-movies.csv')


# ---
# 
# **任务1.2: ** 了解数据
# 
# 你会接触到各种各样的数据表，因此在读取之后，我们有必要通过一些简单的方法，来了解我们数据表是什么样子的。
# 
# 1. 获取数据表的行列，并打印。
# 2. 使用 `.head()`、`.tail()`、`.sample()` 方法，观察、了解数据表的情况。
# 3. 使用 `.dtypes` 属性，来查看各列数据的数据类型。
# 4. 使用 `isnull()` 配合 `.any()` 等方法，来查看各列是否存在空值。
# 5. 使用 `.describe()` 方法，看看数据表中数值型的数据是怎么分布的。
# 
# 

# In[2]:


movie_data.info()


# ---
# 
# **任务1.3: ** 清理数据
# 
# 在真实的工作场景中，数据处理往往是最为费时费力的环节。但是幸运的是，我们提供给大家的 tmdb 数据集非常的「干净」，不需要大家做特别多的数据清洗以及处理工作。在这一步中，你的核心的工作主要是对数据表中的空值进行处理。你可以使用 `.fillna()` 来填补空值，当然也可以使用 `.dropna()` 来丢弃数据表中包含空值的某些行或者列。
# 
# 任务：使用适当的方法来清理空值，并将得到的数据保存。

# In[3]:


#结合 movie_data 缺失值分布的特点，为尽量保证数据的完整性，
#首先删除缺失较为严重且无关紧要的列：'homepage','tagline','keywords','production_companies'
#进而删除轻微缺失的行：‘imdb_id’，‘cast’，‘director’，‘overview’，‘genres’
new_data = movie_data.drop(columns = ['homepage','tagline','keywords','production_companies'])
new_data.dropna(axis = 0, inplace = True)
new_data.shape[0]
#movie_data 中一共10866项，经缺失值处理后剩余10725项，得到了较好的保留


# In[4]:


new_data.head()


# ---
# 
# ---
# 
# ## 第二节 根据指定要求读取数据
# 
# 
# 相比 Excel 等数据分析软件，Pandas 的一大特长在于，能够轻松地基于复杂的逻辑选择合适的数据。因此，如何根据指定的要求，从数据表当获取适当的数据，是使用 Pandas 中非常重要的技能，也是本节重点考察大家的内容。
# 
# 

# ---
# 
# **任务2.1: ** 简单读取
# 
# 1. 读取数据表中名为 `id`、`popularity`、`budget`、`runtime`、`vote_average` 列的数据。
# 2. 读取数据表中前1～20行以及48、49行的数据。
# 3. 读取数据表中第50～60行的 `popularity` 那一列的数据。
# 
# 要求：每一个语句只能用一行代码实现。

# In[5]:


#读取数据表中名为 id、popularity、budget、runtime、vote_average 列的数据。
new_data[['id','popularity','budget','runtime','vote_average']].head(10)


# In[6]:


#读取数据表中前1～20行以及48、49行的数据。
new_data.loc[list(range(20))+[47,48]]


# In[7]:


#读取数据表中第50～60行的 popularity 那一列的数据。
new_data.loc[49:59, 'popularity']#切片，含尾


# # ---
# 
# **任务2.2: **逻辑读取（Logical Indexing）
# 
# 1. 读取数据表中 **`popularity` 大于5** 的所有数据。
# 2. 读取数据表中 **`popularity` 大于5** 的所有数据且**发行年份在1996年之后**的所有数据。
# 
# 提示：Pandas 中的逻辑运算符如 `&`、`|`，分别代表`且`以及`或`。
# 
# 要求：请使用 Logical Indexing实现。

# In[8]:


#读取数据表中 popularity 大于5 的所有数据。
new_data[new_data['popularity'] > 5].head()


# In[9]:


#读取数据表中 popularity 大于5 的所有数据且发行年份在1996年之后的所有数据。
new_data[(new_data['popularity'] >5) & (new_data['release_year'] >= 1996)].head()


# ---
# 
# **任务2.3: **分组读取
# 
# 1. 对 `release_year` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `revenue` 的均值。
# 2. 对 `director` 进行分组，使用 [`.agg`](http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html) 获得 `popularity` 的均值，从高到低排列。
# 
# 要求：使用 `Groupby` 命令实现。

# In[10]:


#对 release_year 进行分组，使用 .agg 获得 revenue 的均值。
new_data.groupby(new_data['release_year'])['revenue'].mean().head()


# In[11]:


#对 director 进行分组，使用 .agg 获得 popularity 的均值，从高到低排列。
new_data.groupby(new_data['director'])['popularity'].mean().sort_values(ascending=False).head()


# ---
# 
# ---
# 
# ## 第三节 绘图与可视化
# 
# 接着你要尝试对你的数据进行图像的绘制以及可视化。这一节最重要的是，你能够选择合适的图像，对特定的可视化目标进行可视化。所谓可视化的目标，是你希望从可视化的过程中，观察到怎样的信息以及变化。例如，观察票房随着时间的变化、哪个导演最受欢迎等。
# 
# <table>
# <thead><tr><th>可视化的目标</th><th>可以使用的图像</th></tr></thead><tbody>
#  <tr><td>表示某一属性数据的分布</td><td>饼图、直方图、散点图</td></tr>
#  <tr><td>表示某一属性数据随着某一个变量变化</td><td>条形图、折线图、热力图</td></tr>
#  <tr><td>比较多个属性的数据之间的关系</td><td>散点图、小提琴图、堆积条形图、堆积折线图</td></tr>
# </tbody></table>
# 
# 在这个部分，你需要根据题目中问题，选择适当的可视化图像进行绘制，并进行相应的分析。对于选做题，他们具有一定的难度，你可以尝试挑战一下～

# **任务3.1：**对 `popularity` 最高的20名电影绘制其 `popularity` 值。

# ---
# **任务3.2：**分析电影净利润（票房-成本）随着年份变化的情况，并简单进行分析。

# In[12]:


movie_data.set_index('original_title')['popularity'].sort_values()[-20:].plot(kind='barh')
plt.xlabel('Popularity')
plt.ylabel('Original Title')
plt.title('Top 20 Movies by Popularity ');


# In[13]:


#票房减成本
new_data['profit'] = new_data['revenue_adj']-new_data['budget_adj']
#以年为分组，统计年平均净利润
new_data.groupby('release_year')['profit'].mean().plot(kind='line', figsize=(20, 10));


# In[14]:


#标准差
new_data.groupby('release_year')['profit'].std().plot(kind='line', figsize=(20, 10));


# In[15]:


#年发行量
new_data.groupby('release_year')['original_title'].count().plot(kind='line', figsize=(20, 10));


# In[16]:




#电影年平均利润在1979年前有较大波动，后逐渐递减，趋于缓和
#随着电影发行数量逐年增加，平均每部电影的利润在减小


# ---
# 
# **[选做]任务3.3：**选择最多产的10位导演（电影数量最多的），绘制他们排行前3的三部电影的票房情况，并简要进行分析。

# In[17]:


#选择最多产的10位导演（电影数量最多的）
#首先我们需要拆分同一部电影的多个导演:
tmp = movie_data['director'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('director') 
movie_data_split = movie_data[['original_title', 'revenue']].join(tmp)
movie_data_split.groupby(movie_data_split['director'])['original_title'].count().sort_values(ascending=False).head(10)


# In[18]:


directors = list(movie_data_split.groupby(movie_data_split['director'])['original_title'].count().sort_values(ascending=False).head(10).index)
f1, f2, f3, f4 = [],[],[],[]
for director in directors:
    #每个导演 top 3 的电影，分别取出电影名称、票房、导演、评价
    a = list(new_data[(new_data['director']== director)]['vote_average'].sort_values(ascending=False)[0:3].rename(index = new_data['original_title']).index)
    b = list(new_data[(new_data['director']== director)]['vote_average'].sort_values(ascending=False)[0:3].rename(index = new_data['revenue_adj']).index)
    c = list(new_data[(new_data['director']== director)]['vote_average'].sort_values(ascending=False)[0:3].rename(index = new_data['director']).index)
    d = list(new_data[(new_data['director']== director)]['vote_average'].sort_values(ascending=False)[0:3].rename(index = new_data['vote_average']).index)
    f1 += a
    f2 += b
    f3 += c
    f4 += d
items = {'director': pd.Series(f3),'original_title': pd.Series(f1), 'revenue_adj': pd.Series(f2),'vote_average': pd.Series(f4)}
df = pd.DataFrame(items)
#将导演设置为索引
df.set_index('director',inplace = True)
df


# In[19]:


#绘制票房与评价散点图
plt.figure(figsize=(15,12))
for i in range(int(len(df)/3)):
    plt.scatter(data = df, x= df['vote_average'][3*i:3*i+3],y = df['revenue_adj'][3*i:3*i+3], s = 200, label = directors[i])
#调整坐标    
plt.yticks(np.linspace(0, 6, 7)*1e8,[str(int(i))+' milion' for i in np.linspace(0,600,7)])
plt.xlabel('vote_average',fontsize = 16)
plt.ylabel('revenue_adj',fontsize = 16)
plt.grid(True)
#绘制图例
plt.legend(loc='upper left')
plt.show()


# In[20]:


df.corr()
#存在叫好又叫座的电影，又存在评价较高的非盈利电影
#Steven Spielberg 导演的电影叫好又叫座，而且是高产导演；
#Martin Scorsese 导演的作品评价较高，但盈利性较差
#Clint Eastwood 导演的作品在评价和盈利上保持了相当稳定的水准
#Ron Howard 导演3部作品评价差别最大
#由相关性可知，票房和评价几乎没有线性相关性


# ---
# 
# **[选做]任务3.4：**分析1968年~2015年六月电影的数量的变化。

# In[21]:


#获取1968年~2015年
sel_year = new_data['release_year'].between(1968,2015,inclusive = True)
#获取6月份
sel_june = list(map(lambda x: (pd.to_datetime(x).month) == 6, new_data['release_date']))

new_data[sel_year&sel_june]['release_year'].value_counts().sort_index().plot(kind='line', figsize=(20, 10), lw = 3);

plt.xlabel('release_year', fontsize = 16);
plt.ylabel('movies_in_June_from_1968_to_2015', fontsize = 16);
plt.grid(True)


# In[22]:


#1968年~2015年六月电影的数量的变化:大体上为上升趋势，短时期内有回落现象，且进入2000后上升趋势加快


# ---
# 
# **[选做]任务3.5：**分析1968年~2015年六月电影 `Comedy` 和 `Drama` 两类电影的数量的变化。

# In[23]:


new_data['genres'].head()


# In[24]:


#选择标签为 Comedy 的
sel_comedy = new_data['genres'].str.contains('Comedy')
#选择标签为 Drama 的
sel_drama = new_data['genres'].str.contains('Drama')
new_data[sel_year&sel_june&sel_comedy]['release_year'].value_counts().sort_index().plot(kind='line', figsize=(20, 10), lw = 3, label = 'Comedy');
new_data[sel_year&sel_june&sel_drama]['release_year'].value_counts().sort_index().plot(kind='line', figsize=(20, 10),lw = 3,label = 'Drama');
plt.grid(True)
plt.xlabel('release_year',fontsize = 16)
plt.ylabel('num_of_type',fontsize = 16)
plt.legend(loc='upper left')
plt.show()


# In[25]:


# Comedy 与 Drama 两类电影的发行量在时间上有较高的相关性


# > 注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出**File -> Download as -> HTML (.html)、Python (.py)** 把导出的 HTML、python文件 和这个 iPython notebook 一起提交给审阅者。
