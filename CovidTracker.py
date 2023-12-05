# install packages
!apt-get install libgeos-dev
!pip install pyproj==1.9.6
!pip uninstall -y shapely 
!pip install shapely --no-binary shapely
!pip install cartopy
%matplotlib inline

# import libraries
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# For improved table display in the notebook
from IPython.display import display

# set printing options for pandas
pd.set_option('max_rows', 2000)

#The first step in our data science model in the week 9 lectures, is to collect or identify data. I have provided for you a direct link to the COVID-19 data on Google drive. 
#We can access it by treating the file as a StringIO object and passing that into the pandas read_csv() function.
data_url = requests.get('https://drive.google.com/uc?export=download&id=1pRnqYs1nuBEbwUJAuQApmyWETSYEVWDy')
csv_raw = StringIO(data_url.text)
covid19_data = pd.read_csv(csv_raw,low_memory=False)

# You can extract values from a data frame in many different ways. 
# To retrieve a column we can use   df[colname]
print(covid19_data['country'].head(n=3)) # here we use head simply to suppress the large amount of output

# We can also use this syntax
print(covid19_data.country.head(n=3)) 

# or pass a list to get multiple columns
print(covid19_data[['country','province']].head(n=3)) 

#.loc() can be used to set a range of rows and/or columns (by name)
print(covid19_data.loc[10:15,['country','province']])

#.iloc() can be used to set a range of rows and/or columns (by index)
print(covid19_data.iloc[10:15,0:4])

#Prob1. 1Write a function named rows_and_columns that takes in a pandas data frame and returns the string:The data has X rows and Y columns. 
#where X is the number of rows and Y is the number of columns. For example, if the data frame has 100 rows and 10 columns, the function should return the string: The data has 100 rows and 10 columns.
def rows_and_columns(dataframe):
 
    rowsandcols = dataframe.shape
    totalsize = dataframe.size
    
    row = rowsandcols[0]
    col = rowsandcols[1]
    
    finalnumber = "The data has " + str(row) + " rows and " + str(col) + " columns."
    return finalnumber
 
    
# Prob 1.2 Write a function named get_min_max that takes in a pandas data frame and a column name as a string, and returns the minimum and maximum value of that column in a tuple
def get_min_max(dataframe, colname):
    min_max = ()
    minval = dataframe[::1][colname].min()
    maxval = dataframe[::1][colname].max()
 
    min_max = min_max + (minval,)
    min_max = min_max + (maxval,)
    return min_max
 
# PRob 1.3 Write a function named odd_get_min_max that takes in a pandas data frame and a column name as a string, and returns the minimum and maximum values for the odd rows and that column in a tuple
def odd_get_min_max(dataframe, col_name):
 
    odd_rows = dataframe[col_name].iloc[1::2]
 
    minval = odd_rows.min()
 
    maxval = odd_rows.max()
 
    return minval, maxval

# here we call your functions
print(rows_and_columns(covid19_data))

print(get_min_max(covid19_data,'latitude'))

print(odd_get_min_max(covid19_data,'latitude'))

print(covid19_data.columns)
print(covid19_data.head())


#dataframe info() function is a great way to get a summary of the input data.
covid19_data.info()

#info() shows us that most columns have significantly high levels of missing data. Typically, columns with high levels of missing data are removed or imputed. 
#Here, we will ignore the missing data. The describe() function is more useful when you have numerical data, but it still provides useful information on how our data are distributed.

covid19_data.describe(include="all")


#The data are messy. Various parties have contibuted to the dataset without following a consistent formatting for the columns. 
#If we are interested in questions about age, for example, we need to clean the age column. First, let's visualize the age column data by counting the unique fields.
# Probem 2) Write a function named "get_uniq" that takes in a pandas data frame and a column name, and returns a numpy ndarray containing the unique values in that column.
def get_uniq(dataframe, colname):
  return dataframe[colname].unique()

print(get_uniq(covid19_data,'age'))

print(covid19_data['age'].value_counts())


# Problem 3) Define a function named "unique_nonNaN_cnt" that takes a pandas data frame, a column name as a string, 
#and returns the number of unique non-NaN values. You can think about this as either counting the non-NaN values or summing up the unique non-NaN values from the value_counts() method.
def unique_nonNaN_cnt(dataframe, colname):
  return sum(dataframe[colname].value_counts())
 
 print("Total of " + str(unique_nonNaN_cnt(covid19_data,'age')) + " non-NaN age entries.")
#There is a large amount of missing data, and a large variety of entries. We should clean the age columns.
#Let's convert the ages to age ranges for plotting. For the existing ranges in the data, let's consider the mean age.
# cleaning the age column
# We observe that the age column does not follow a nice format

# defining the age ranges
age_ranges = []
for age in range(0,100,10):
  age_ranges.append((age,age+10))
print("Considering age ranges",age_ranges)

# helper function that takes in an numerical age, and a list of ranges and 
# returns the particular range that the age falls into
def findRange(age,arange):
  for ager in arange:
    if age >= ager[0] and age < ager[1]:
      return str(ager[0])+"-"+str(ager[1])

# a function that will fix our age entries
def fixAge(age):
  if isinstance(age, str): # check if the age is a string
    if 'weeks' in age:
      age = age.replace('weeks','')
      age = str(float(age)/52.0)
    if '-' in age: # if the string has a hyphen (is a range)
      parts = age.split("-")
      if len(parts)==1: # if the range was poorly formatted...
        return findRange(float(parts[0]),age_ranges)
      elif len(parts)==2: # if the range was properly formatted, findRange of the mean
        if parts[1]=='':
          return findRange(float(parts[0]),age_ranges)  
        else:
          return findRange(np.mean([float(parts[0]),float(parts[1])]),age_ranges)
      else:
        print(age)
        raise InputError("Age " + str(age) + " not correctly handled.")
    else: 
        return findRange(float(age),age_ranges)
  elif np.isnan(age):
    return np.nan
  else:
    raise InputError("Age " + str(age) + " not correctly handled.")

# create a new column that holds the new age ranges
# this code will run the fixAge function for each row of the covid data frame and
# store the result in the newly created 'age_range' column
covid19_data['age_range'] = covid19_data.apply(lambda row : fixAge(row['age']), axis = 1) 

#apply()
print("the total number of rows with non-NaN ages is " + str(sum(covid19_data['age'].value_counts())))
print("the total number of rows with non-NaN age_ranges is " + str(sum(covid19_data['age_range'].value_counts())))


# distribution of cases by age
age_range_labels = [str(x[0])+"-"+str(x[1]) for x in age_ranges]
counts = covid19_data.age_range.value_counts()[age_range_labels]

# create plot
fig, ax = plt.subplots(figsize=(20, 10))
index = np.arange(len(age_ranges))
bar_width = 0.35
opacity = 0.8

# docs are here: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.bar.html
rects1 = plt.bar(index, counts, bar_width,alpha=opacity,color='b')

plt.xlabel('Age Range')
plt.ylabel('Count')
plt.title('Corona Cases per Age Group')
plt.xticks(index, ["["+str(x[0])+","+str(x[1])+")" for x in age_ranges])
plt.legend()

plt.tight_layout()


# distribution of cases by age and sex
# Problem 4) Complete where we have indicated below
def create_bar_plot_by_sex(covid19_data, age_ranges):
  age_range_labels = [str(x[0])+"-"+str(x[1]) for x in age_ranges]
  # from the covid19_data, select the age_range for female rows 
  female_age_ranges = covid19_data[covid19_data['sex']=='female']['age_range'] # problem 4, fill this in
  counts_female = female_age_ranges.value_counts()[age_range_labels]
  
  # from the covid19_data, select the age_range for male rows 
  male_age_ranges = covid19_data[covid19_data['sex']=='male']['age_range'] # problem 4, fill this in
  counts_male = male_age_ranges.value_counts()[age_range_labels]
 
  # create plot
  fig, ax = plt.subplots(figsize=(20, 10))
  index = np.arange(len(age_ranges))
  bar_width = 0.35
  opacity = 0.8
 
  # the bar function draws a bar plot, the first two arugments are the x position of the bar, and its height
  rects1 = plt.bar(index,counts_male , bar_width, # problem 4, fill in first two arguments
                  alpha=opacity,color='b',label='Male')
 
  rects2 = plt.bar(index,counts_female , bar_width, # problem 4, fill in first two arguments hint: you have to use the bar_width in the first argument
                  alpha=opacity,color='g',label='Female')
 
  plt.xlabel('Age Range')
  plt.ylabel('Count')
  plt.title('Corona Cases per Age Group')
  #plt.xticks(index + bar_width, age_ranges)
  plt.xticks(index, ["["+str(x[0])+","+str(x[1])+")" for x in age_ranges])
  plt.legend()
 
  plt.tight_layout()
  return counts_female, counts_male
 cnts_f, cnts_m = create_bar_plot_by_sex(covid19_data, age_ranges)

# distribution of cases by country
def create_bar_plot_by_country(covid19_data):
  country_cnts = covid19_data.country.value_counts()

  n_groups = len(country_cnts)
  counts = country_cnts

  # create plot
  fig, ax = plt.subplots(figsize=(20, 10))
  index = np.arange(n_groups)
  bar_width = 0.35
  opacity = 0.8

  rects1 = plt.bar(index, counts, bar_width,
                  alpha=opacity,color='b')

  plt.xlabel('Country')
  plt.ylabel('Count')
  plt.title('Corona Cases per Country')
  #plt.xticks(index + bar_width, age_ranges)
  plt.xticks(index, country_cnts.index.values)
  plt.legend()

  plt.tight_layout()
  return n_groups, counts
ngrps, cnts = create_bar_plot_by_country(covid19_data)


# distribution of cases by country with >1000 cases
# Problem 5) Print the same bar plot by country, but limit the plot to countries that have >1000 cases.
def create_bar_plot_by_country(covid19_data):
  country_cnts = covid19_data.country.value_counts()
 
  # get the counts for countries with >1000 cases, this should be a data series
  counts =   country_cnts[country_cnts > 1000]
  # get number of countries with >1000 cases, this should be an integer
  n_groups = len(country_cnts[country_cnts > 1000])
 
  # create plot
  fig, ax = plt.subplots(figsize=(20, 10))
  index = np.arange(n_groups)
  bar_width = 0.35
  opacity = 0.8
 
  rects1 = plt.bar(index, counts, bar_width,
                  alpha=opacity,color='b')
 
  plt.xlabel('Country')
  plt.ylabel('Count')
  plt.title('Corona Cases per Country')
  plt.xticks(index, country_cnts.index.values ) # Problem 5, fill this in
  plt.legend()
 
  plt.tight_layout()
  return n_groups, counts
  ngrps, cnts = create_bar_plot_by_country(covid19_data)


#groupby() method
map_intensities = covid19_data.groupby([covid19_data.latitude.round(1), 
                                        covid19_data.longitude.round(1), 
                                        covid19_data.country]).ID.count().reset_index()'

# cases across the globe using cartopy
# set the colors for countries
map_intensities['labels_enc'] = pd.factorize(map_intensities['country'])[0] 

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())

ax.set_global()

ax.add_feature(cfeature.COASTLINE, edgecolor="tomato")
ax.add_feature(cfeature.BORDERS, edgecolor="tomato")
ax.gridlines()

plt.scatter(x=map_intensities['longitude'], y= map_intensities['latitude'],
            s=map_intensities['ID']/10,
            c=map_intensities['labels_enc'],
            cmap="Set1",
            alpha=0.4,
            transform=ccrs.PlateCarree()) 


#morality?
print(covid19_data['outcome'].value_counts())

pos=['discharge','stable','discharged','recovered','stable condition','Alive','Stable','released from quarantine','Recovered','Discharged from hospital','Discharged']
neg=['died','death','Dead','severe','critical condition, intubated as of 14.02.2020','dead','Death','Deceased','severe illness','unstable','Died','Critical condition']

def setOutcomeClass(outcome):
  if outcome in pos:
    return 1
  elif outcome in neg:
    return 0
  else:
    return np.nan

covid19_data['outcome_class'] = covid19_data.apply(lambda row : setOutcomeClass(row['outcome']), axis = 1) 

#linegraph.  across age groups, but each age group can have a different number of samples. 
#Therefore, we compute an empirical probability of a positive outcome but also include  ±  1 standard deviation. We also include Spearman's correlation on the plot.
# subset the data by age range and outcome class, then group by age range,
# and use the agg (aggregate) function to compute the mean, count, and 
# standard deviation by age group
outcomes_per_age = covid19_data[['age_range','outcome_class']].groupby(['age_range']).agg(['mean','count','std']).reset_index()
x = outcomes_per_age.age_range
y = outcomes_per_age.outcome_class['mean']
error = outcomes_per_age.outcome_class['std']

fig, ax = plt.subplots(figsize=(20, 10))
ax.errorbar(x, y, yerr=error, fmt='-o')
plt.ylabel('Relative Frequency', fontsize=14)
plt.xlabel('Age Group', fontsize=14)

fig.text(0.2,0.2,"spearman correlation = " + str(covid19_data['age_range'].corr(covid19_data['outcome'],method='spearman')), fontsize=14)

  # Problem 6) Professor Derek is worried about outcomes over time for his age bracket (30-40). He wants you to plot the relative frequency of positive outcomes (y-axis) over time (x-axis) 
#while also including 1 standard deviation above and below each point. You should not compute Spearman's correlation here. Fill in the function below.


def create_bar_plot_for_derek(covid19_data):
  # first we subset the data by the appropriate age bracket and do a bit of cleaning
  prof_age_data = covid19_data[covid19_data['age_range'] =="30-40"]
  prof_age_data=prof_age_data.replace(to_replace='25.02.2020 - 26.02.2020',value='25.02.2020')
 
  # and we convert the column to a date-time
  prof_age_data['date_confirmation']=pd.to_datetime(prof_age_data['date_confirmation'],dayfirst=True)
 
  outcomes_over_time = prof_age_data[['date_confirmation', 'outcome_class']].groupby(['date_confirmation']).agg(['mean','count','std']).reset_index()
 
  outcomes_over_time = outcomes_over_time.dropna() # we should drop the rows with missing values
 
  x = outcomes_over_time.date_confirmation
  y =  outcomes_over_time.outcome_class['mean']
  error =  outcomes_over_time.outcome_class['std']
 
  fig, ax = plt.subplots(figsize=(20, 10))
  ax.errorbar(x, y, yerr=error, fmt='-o')
  plt.ylabel('Relative Frequency', fontsize=14)
  plt.xlabel('Date', fontsize=14)
  return x, y, error

x,y,error = create_bar_plot_for_derek(covid19_data)

#Are reported cases of COVID-19 more prevalant in colder climates?
# latitude data ranges from -90 (south pole) to 90 (north pole)
print(covid19_data['latitude'].describe())
fig, ax = plt.subplots(figsize=(20, 10))

num_bins = 90

# the histogram of the data
ax.hist(abs(covid19_data['latitude']), num_bins, density=1,alpha=0.3)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Absolute Latitude Degree', fontsize=14)


#latitudes 20 and 60
# I downloaded and prepared the data for you.
population_data_url = requests.get('https://drive.google.com/uc?export=download&id=19BjvYrh_MkzE2NMJBOSJzJUaXaw3S85X')
population_csv = StringIO(population_data_url.text)
population_data = pd.read_csv(population_csv, delimiter=" ", header=None)
abs_latitude=np.linspace(0,90,360)

# population data goes from 90 degrees to -90 degrees in increments of 0.25 degrees
lat_sums=np.sum(population_data,axis=1)
lat_by_degree = lat_sums.groupby(np.arange(len(lat_sums))//4).sum()
population_sums = lat_by_degree.groupby(np.concatenate((np.arange(0,len(lat_by_degree)/2),np.arange(0,len(lat_by_degree)/2)[::-1]))).sum()

fig, ax = plt.subplots(figsize=(20, 10))

num_bins = 90

# the histogram of the data
ax.hist(abs(covid19_data['latitude']), num_bins, density=1,alpha=0.3)
ax.hist(range(num_bins)[::-1],bins=num_bins, density=1, weights=population_sums,alpha=0.3)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Absolute Latitude Degree', fontsize=14)

#We see that there are a considerable number of people who live close to the equator (latitude=0) so infections indeed are more prevalant
