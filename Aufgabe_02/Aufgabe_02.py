import numpy as np
import matplotlib.pyplot as plt

d = np.genfromtxt("Aufgabe_02/data/london_weather.csv", delimiter=",", skip_header=1)

dt =  d[:,0]
day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')

temp = d[:,5]

temp1980 = temp[year == 1980]
temp1990 = temp[year == 1990]
temp2000 = temp[year == 2000]
temp2010 = temp[year == 2010]

#1.1

# plt.boxplot([temp1980, temp1990, temp2000, temp2010])
# plt.show()

#1.2

# plt.scatter(day[year == 2010], temp2010)
# plt.show()

#1.3

# low1980 = np.quantile(temp1980, 0.05)
# high1980 = np.quantile(temp1980, 0.95)
# low2010 = np.quantile(temp2010, 0.05)
# high2010 = np.quantile(temp2010, 0.95)

# print("1980: ", low1980, high1980)
# print("2020: ", low2010, high2010)

#1.4

# for i in range(2010, 2021):
#     temp_year = temp[year == i]
#     median = np.median(temp_year)
#     plt.bar(i, median, color='blue')
    
# plt.show()

#1.5

for i in range(1, 13):
    temp_month = temp[(year == 2000) & (month == i)]
    median = np.median(temp_month)
    plt.bar(i, median, color='blue')
    
plt.show()
