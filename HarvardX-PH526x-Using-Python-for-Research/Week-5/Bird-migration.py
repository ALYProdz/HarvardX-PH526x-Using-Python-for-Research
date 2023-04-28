#Let's import the module
import pandas as pd
import datetime
import matplotlib.pyplot as plt
#Import the Data for the analyze
Birddata = pd.read_csv("./week-5/bird_tracking.csv", index_col=0)
#Find the infomation about the data that we have to understand it.
"""
Birddata.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 61920 entries, 0 to 61919
Data columns (total 8 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   altitude            61920 non-null  int64  
 1   date_time           61920 non-null  object 
 2   device_info_serial  61920 non-null  int64  
 3   direction           61477 non-null  float64
 4   latitude            61920 non-null  float64
 5   longitude           61920 non-null  float64
 6   speed_2d            61477 non-null  float64
 7   bird_name           61920 non-null  object 
dtypes: float64(4), int64(2), object(2)
memory usage: 4.3+ MB
"""

#Let's groupig the data by name
grouped_bird = Birddata.groupby("bird_name")
# Now calculate the mean of `speed_2d` using the `mean()` function.
mean_speeds = grouped_bird.speed_2d.mean()
"""
The result of the mean_speed by bird_name
bird_name
Eric     2.300545
Nico     2.908726
Sanne    2.450434"""
# Use the `head()` method prints the first 5 lines of each bird.
grouped_bird.head()
"""
 altitude               date_time  ...   speed_2d  bird_name
0            71  2013-08-15 00:18:08+00  ...   0.150000       Eric
1            68  2013-08-15 00:48:07+00  ...   2.438360       Eric
2            68  2013-08-15 01:17:58+00  ...   0.596657       Eric
3            73  2013-08-15 01:47:51+00  ...   0.310161       Eric
4            69  2013-08-15 02:17:42+00  ...   0.193132       Eric
"""
# Find the mean `altitude` for each bird.
mean_altitude = grouped_bird.altitude.mean()
"""
mean_altitude
Out[38]: 
bird_name
Eric     60.249406
Nico     67.900478
Sanne    29.159922
Name: altitude, dtype: float64
"""
# Convert birddata.date_time to the `pd.datetime` format.
Birddata.date_time = pd.to_datetime(Birddata.date_time)
"""
Birddata.date_time
Out[40]: 
0       2013-08-15 00:18:08+00:00
1       2013-08-15 00:48:07+00:00
2       2013-08-15 01:17:58+00:00
3       2013-08-15 01:47:51+00:00
4       2013-08-15 02:17:42+00:00
"""
# Create a new column of day of observation
Birddata["date"] = Birddata.date_time.dt.date
"""
Birddata["date"]
Out[44]: 
0        2013-08-15
1        2013-08-15
2        2013-08-15
3        2013-08-15
4        2013-08-15
"""
# Check the head of the column.
Birddata["date"].head()
Birddata.date.head()
"""
Birddata.date.head()
Out[48]: 
0    2013-08-15
1    2013-08-15
2    2013-08-15
3    2013-08-15
4    2013-08-15
Name: date, dtype: object
"""
# Use `groupby()` to group the data by date.
grouped_by_date = Birddata.groupby("date")
"""
grouped_by_date.head()
Out[52]: 
       altitude                 date_time  ...  bird_name        date
0            71 2013-08-15 00:18:08+00:00  ...       Eric  2013-08-15
1            68 2013-08-15 00:48:07+00:00  ...       Eric  2013-08-15
2            68 2013-08-15 01:17:58+00:00  ...       Eric  2013-08-15
3            73 2013-08-15 01:47:51+00:00  ...       Eric  2013-08-15
4            69 2013-08-15 02:17:42+00:00  ...       Eric  2013-08-15
"""

# Find the mean `altitude` for each date.
mean_altitude_perday = grouped_by_date.altitude.mean()
"""
mean_altitude_perday
Out[54]: 
date
2013-08-15    134.092000
2013-08-16    134.839506
2013-08-17    147.439024
2013-08-18    129.608163
2013-08-19    180.174797
   
2014-04-26     15.118012
2014-04-27     23.897297
2014-04-28     37.716867
2014-04-29     19.244792
2014-04-30     13.954545
Name: altitude, Length: 259, dtype: float64
"""
# Use `groupby()` to group the data by bird and date.
grouped_birdday = Birddata.groupby(["bird_name", "date"])
"""
grouped_birdday.head()
Out[57]: 
       altitude                 date_time  ...  bird_name        date
0            71 2013-08-15 00:18:08+00:00  ...       Eric  2013-08-15
1            68 2013-08-15 00:48:07+00:00  ...       Eric  2013-08-15
2            68 2013-08-15 01:17:58+00:00  ...       Eric  2013-08-15
3            73 2013-08-15 01:47:51+00:00  ...       Eric  2013-08-15
4            69 2013-08-15 02:17:42+00:00  ...       Eric  2013-08-15
"""

# Find the mean `altitude` for each bird and date.
mean_altitude_grouped_birdday = grouped_birdday.altitude.mean()
"""
mean_altitude_grouped_birdday
Out[59]: 
bird_name  date      
Eric       2013-08-15     74.988095
           2013-08-16    127.773810
           2013-08-17    125.890244
           2013-08-18    121.353659
           2013-08-19    134.928571
   
Sanne      2014-04-26     17.116667
           2014-04-27     17.391892
           2014-04-28     58.876712
           2014-04-29     30.530120
           2014-04-30      4.361111
Name: altitude, Length: 770, dtype: float64
"""
# look at the head of `mean_altitudes_perday`.
mean_altitude_perday.head()
"""
mean_altitude_perday.head()
Out[62]: 
date
2013-08-15    134.092000
2013-08-16    134.839506
2013-08-17    147.439024
2013-08-18    129.608163
2013-08-19    180.174797
Name: altitude, dtype: float64
"""

#Some Simple visualization of the Bird
lx = Birddata.bird_name == "Eric"
x , y = Birddata.longitude[lx] , Birddata.latitude[lx]
ls = Birddata.bird_name == "Sanne"
c , d = Birddata.longitude[ls] , Birddata.latitude[ls]

plt.figure()
plt.plot(c,d,".", )
plt.title("Longitude and LAtitude of Sanne Bird, kwargs")
plt.xlabel("longitude")
plt.ylabel("Latitude")


plt.figure()
plt.plot(x,y,".", )
plt.title("Longitude and Latitude of Eric Bird, kwargs")
plt.xlabel("longitude")
plt.ylabel("Latitude")


