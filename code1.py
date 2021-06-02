import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import datetime
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import time


###################################################################

covid = pd.read_csv("covid.csv")
CovidCases = 0
for i, row in covid.iterrows():
	if row['Date_reported'] == '2020-08-01':
		CovidCases = CovidCases + row[' Cumulative_cases']

print("Covid-19 cases: ",CovidCases)

Coviddeaths = 0
for i, row in covid.iterrows():
	if row['Date_reported'] == '2020-08-05':
		Coviddeaths = Coviddeaths + row[' Cumulative_deaths']

print("Covid-19 deaths: ",Coviddeaths)

mortco = (Coviddeaths/CovidCases)*100
print('Covid-19 mortality rate: ', str(mortco)+"%","\n")

covidNewCases = covid[['Date_reported', ' New_cases']]
covidNewCases = covidNewCases.groupby('Date_reported').sum()

covidNewDeaths = covid[['Date_reported', ' New_deaths']]
covidNewDeaths = covidNewDeaths.groupby('Date_reported').sum()


covidtotalCases = covid[['Date_reported' , ' Cumulative_cases']]
covidtotalCases = covidtotalCases.groupby('Date_reported').max()

covidtotalDeaths = covid[['Date_reported' , ' Cumulative_deaths']]
covidtotalDeaths = covidtotalCases.groupby('Date_reported').max()

covidRegions = covid[[' WHO_region' , ' Cumulative_cases']]
covidRegions = covidRegions.groupby(' WHO_region').max()
covidRegions.to_csv("CovidRegions.csv")

conumbercountry = covid[[' Country']].groupby(' Country').count()
# print('Number of countries affected by Covid-19: ', conumbercountry)


###################################################################

H1N1 = pd.read_csv("Pandemic (H1N1) 2009.csv")
H1N1gt = pd.read_csv("Grandtotals.csv")

h1cases = H1N1[[ 'Cases']]
h1casestotal_unformatted = h1cases.max()
h1casestotal = int(h1casestotal_unformatted)
print("H1N1 cases: ",h1casestotal)

h1deaths = H1N1[[ 'Deaths']]
h1deathstotal_unformatted = h1deaths.max()
h1deathstotal = int(h1deathstotal_unformatted)
print("H1N1 deaths: ",h1deathstotal)

morth1 = (h1deathstotal/h1casestotal)*100
print('H1N1 mortality rate: ', str(morth1)+"%","\n")

tempNewcases = H1N1gt[['Country', 'Cases', 'Update Time']]

h1gt = tempNewcases.sort_values(by=['Cases']) #ascending=False
temp = h1gt['Cases']

tempNewdeaths = H1N1gt[['Country', 'Deaths', 'Update Time']]
h1gt = tempNewdeaths.sort_values(by=['Deaths']) #ascending=False
temp1 = h1gt['Deaths']

#print(temp)

gtcases = temp.values.tolist()
gtdeaths = temp1.values.tolist()
#print(gtcases)

#[x - gtcases[i - 1] for i, x in enumerate(gtcases)][1:]

h1n1NewCases = np.diff(gtcases)
h1n1Newdeaths = np.diff(gtdeaths)

#np.savetxt("H1N1New_cases.csv", h1n1NewCases, delimiter=",")

h1n1_casesbycountry = H1N1[['Country', 'Cases']]
h1n1countryc = h1n1_casesbycountry.groupby('Country').max()
h1n1countryc.to_csv('H1N1_casesbyCountry.csv')

h1n1_deathsbycountry = H1N1[['Country', 'Deaths']]
h1n1countryd = h1n1_deathsbycountry.groupby('Country').max()
h1n1countryd.to_csv('H1N1_deathsbyCountry.csv')

sfnumbercountry = H1N1[['Country']].groupby('Country').count()
print('Number of countries affected by swineflue: ', sfnumbercountry)

# def junk():
# 	h1country = h1cases.groupby('Country')
# 	maxh1 =  h1country.max()
# 	h1casestotal_unformatted = maxh1.sum()
# 	h1casestotal = int(h1casestotal_unformatted)
# 	print("H1N1 cases: ", h1casestotal)

# 	h1deaths = H1N1[['Country', 'Deaths']]
# 	h1country = h1deaths.groupby('Country')
# 	maxh1 =  h1country.max()
# 	h1deathstotal_unformatted = maxh1.sum()
# 	h1deathstotal = int(h1deathstotal_unformatted)
# 	print("H1N1 deaths: ", h1deathstotal)

# 	morth1 = (h1deathstotal/h1casestotal)*100
# 	print('H1N1 mortality rate: ', str(morth1)+"%","\n")

###################################################################

ebola = pd.read_csv("ebola_2014_2016_clean.csv")

ebcases = ebola[[ 'Country', 'Cumulative no. of confirmed, probable and suspected cases']]
ebcountry = ebcases.groupby('Country')

maxeb =  ebcountry.max()
ebcasestotal_unformatted = maxeb.sum()
ebcasestotal = int(ebcasestotal_unformatted)

print("Ebola cases: ", ebcasestotal)


ebdeaths = ebola[[ 'Country', 'Cumulative no. of confirmed, probable and suspected deaths']]
ebcountry = ebdeaths.groupby('Country')

maxeb =  ebcountry.max()
ebdeathstotal_unformatted = maxeb.sum()
ebdeathstotal = int(ebdeathstotal_unformatted)

print("Ebola deaths: ", ebdeathstotal)

morteb = (ebdeathstotal/ebcasestotal)*100
print('ebola mortality rate: ', str(morteb)+"%","\n")

ebola_newcases = ebola[['Date', 'Cumulative no. of confirmed, probable and suspected cases']]
ebola_dates = ebola_newcases.groupby('Date')
ebola_casetotals = ebola_dates.sum()
eboladialycases = ebola_casetotals['Cumulative no. of confirmed, probable and suspected cases']

ebdialycaseList = eboladialycases.sort_values()
#ebdialycaseList.to_csv('ebolacases.csv')
ebdialycaseList = eboladialycases.values.tolist()
#print(ebdialycaseList)


ebola_newdeaths = ebola[['Date', 'Cumulative no. of confirmed, probable and suspected deaths']]
ebola_dates = ebola_newdeaths.groupby('Date')
ebola_deathtotals = ebola_dates.sum()
eboladialydeaths = ebola_deathtotals['Cumulative no. of confirmed, probable and suspected deaths']

ebdialydeathList = eboladialydeaths.sort_values()
ebdialydeathList = eboladialydeaths.values.tolist()
# print(ebdialydeathList)

EbolaNewCases = np.diff(ebdialycaseList) 
EbolaNewdeaths = np.diff(ebdialydeathList)

#np.savetxt("EbolaNew_cases.csv", EbolaNewCases, delimiter=",")
#EbolaNewCases.to_csv('EbolaNew_cases.csv')

ebola_casesbycountry = ebola[['Country', 'Cumulative no. of confirmed, probable and suspected cases']]
ebolacountryc = ebola_casesbycountry.groupby('Country').max()
ebolacountryc.to_csv('Ebola_casesbyCountry.csv')

ebola_deathsbycountry = ebola[['Country', 'Cumulative no. of confirmed, probable and suspected deaths']]
ebolacountryd = ebola_deathsbycountry.groupby('Country').max()
ebolacountryd.to_csv('Ebola_deathsbyCountry.csv')

#EbolaNewdeaths.to_csv('EbolaNewdeaths.csv')

# print(EbolaNewCases)

###################################################################

objects = ('H1N1', 'Ebola','Covid-19')
y_pos = np.arange(len(objects))
performance = [h1casestotal,ebcasestotal,CovidCases]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Cumulative numbers')
plt.title('TOTAL CASES')
plt.savefig('TOTAL_CASES_COMPARISON.png')
plt.clf()



plt.plot(covidNewCases, '-o')
plt.xlabel('Time')
plt.ylabel('New Cases')
plt.savefig('CovidCases.png')
plt.clf()

plt.plot(covidNewDeaths, '-o')
plt.xlabel('Time')
plt.ylabel('New Deaths')
plt.savefig('Coviddeaths.png')
plt.clf()

plt.figure(figsize=(16, 9))
plt.plot(covidNewCases,label='Cases')
plt.plot(covidNewDeaths, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Numbers')
plt.title('Daily Cases vs Deaths')
plt.legend()
plt.savefig('coviddailycasesvsdeaths.png')

plt.plot(covidtotalCases)
plt.xlabel('Time')
plt.ylabel('Cases')
plt.title('Covid-19 Total Cases')
plt.savefig('Covid-19TotalCases.png')
plt.clf()

plt.plot(covidtotalDeaths)
plt.xlabel('Time')
plt.ylabel('Deaths')
plt.title('Covid-19 Total Deaths')
plt.savefig('Covid-19TotalDeaths.png')
plt.clf()



plt.plot(h1n1NewCases)
plt.savefig('h1n1_New_Cases.png')
plt.clf()

plt.plot(h1n1Newdeaths)
plt.savefig('h1n1_New_Deaths.png')
plt.clf()

plt.figure(figsize=(16, 9))
plt.plot(h1n1NewCases,label='Cases')
plt.plot(h1n1Newdeaths, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Numbers')
plt.title('Daily Cases vs Deaths')
plt.legend()
plt.savefig('swinefludailycasesvsdeaths.png')

plt.plot(EbolaNewCases)
plt.savefig('Ebola_New_Cases.png')
plt.clf()

plt.plot(EbolaNewdeaths)
plt.savefig('Ebola_New_Deaths.png')
plt.clf()

plt.figure(figsize=(16, 9))
plt.plot(EbolaNewCases,label='Cases')
plt.plot(EbolaNewdeaths, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Numbers')
plt.title('Daily Cases vs Deaths')
plt.legend()
plt.savefig('eboladailycasesvsdeaths.png')

x = pd.read_csv('covidNewCases.csv')
y = pd.read_csv('covidNewDs.csv')

plt.figure(figsize=(16, 9))
plt.plot(x,label='Cases')
plt.plot(y, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Numbers')
plt.title('Daily Cases vs Deaths')
plt.legend()
plt.savefig('Regression.png')

plt.clf()

#plt.show()

##################################################################


# cossim = pd.DataFrame(np.random.randint(0, 2, 2))
# cosine_similarity(existing_set)


# B = pd.read_csv('EbolaNew_cases.csv')
# C = pd.read_csv('H1N1New_cases.csv')


