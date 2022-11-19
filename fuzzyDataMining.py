# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:34:03 2022

@author: rcarley
"""
import numpy as np
import pandas as pd
import scipy.stats as st

#import skfuzzy as 

phrDF = pd.read_csv('C:\\Users\\rcarley\\Downloads\\phr.csv')
print (phrDF.columns)
phrStats = phrDF.describe()
phrMedian = phrDF.median()

binaryVariables = ['Female', 'Caucasian', 'Insurance','Eye Exam', 'Pneumo Vaccine', 'ACE ARB ALB', 'Foot Exam', 'Nonsmoker', 'HBA1C Test' ]
continuousVariables = ['Age', 'Income', 'Engagement', 'Logins', 'HBA1C', 'SBP', 'DBP', 'LDL', 'BMI']
'''
print ("out of", len(phrDF) )
for i in binaryVariables:
    sum1 = phrDF[i].sum()
    print( i,":", sum1, ",", str(format(sum1/len(phrDF), '.2f')) + "%" )
    
print("------------------------------")    
for i in continuousVariables:
    print(phrStats[i] )
'''   

import skfuzzy as fuzz


#create fuzzy sets---------------------------------------


#Age 
ageMin = phrStats['Age']['min']
ageMax = phrStats['Age']['max']
age25 = phrStats['Age']['25%']
age75 = phrStats['Age']['75%']
age50 = phrStats['Age']['50%']
ageX = np.arange(phrStats['Age']['min'], phrStats['Age']['max'])

ageLowX = fuzz.trapmf(ageX, [-1, 0, 25, 40]) #young
ageMidX = fuzz.trapmf(ageX, [25, 40, 50, 60]) #middle
ageHighX = fuzz.trapmf(ageX, [50, 60, 100, 130]) #old

#medan income based on Cuyahoga county Ohio census data
#using BLS inflation calculator to get january 2010 to 
#january 2020 inflation calculation, since the
#census numbers were given in 2020 dollars.
#https://www.bls.gov/data/inflation_calculator.htm
#https://www.census.gov/quickfacts/fact/table/cuyahogacountyohio,clevelandcityohio/POP010210

medianIncome = phrStats['Income']['50%'] #43.46071
incomeX = np.arange(phrStats['Income']['min'], phrStats['Income']['max'])


incomeLowX = fuzz.trapmf(incomeX, [-1,20000, 30000, 40000]) #poor
incomeMidX = fuzz.trapmf(incomeX, [30000,40000 , 50000, 60000]) #middle class
incomeHighX = fuzz.trapmf(incomeX, [50000, 60000, 124000, 124001]) #well off


#both of thes will be calculated using stats\
#engagement
enMin = phrStats['Engagement']['min']
enMax = phrStats['Engagement']['max']
en25 = phrStats['Engagement']['25%']
en75 = phrStats['Engagement']['75%']
en50 = phrStats['Engagement']['50%']
enTen = (enMax - enMin) * .1
enX = np.arange(enMin, enMax+ .1, 0.1)

enLowX = fuzz.trapmf(enX, [enMin-.1, enMin, en25,en50])
#enMiddleX = fuzz.trapmf(enX, [en25, en50 - enTen , en50 + enTen, en75])
enHighX = fuzz.trapmf(enX, [en50, en75, enMax+.1,enMax +.2 ])

#logins 
logMin = phrStats['Logins']['min']
logMax = phrStats['Logins']['max']
log25 = phrStats['Logins']['25%']
log75 = phrStats['Logins']['75%']
log50 = phrStats['Logins']['50%']
logTen = (logMax - logMin) * .1
logX = np.arange((logMax-logMin))

logLowX = fuzz.trapmf(logX, [logMin-1, logMin, log25,log50])
#logMiddleX = fuzz.trapmf(x, [log25, log50 - ten , log50 + ten, en75])
logHighX = fuzz.trapmf(logX, [log50, log75, logMax+1,logMax +2 ])

#HBA1C
'''
#Normal	Below 42 mmol/mol	Below 6.0%
#Prediabetes	42 to 47 mmol/mol	6.0% to 6.4%
#Diabetes	48 mmol/mol or over	6.5% or over
#https://www.diabetes.co.uk/what-is-hba1c.html
'''
hbX = np.arange(phrStats['HBA1C']['min'], phrStats['HBA1C']['max'])
hbNormalX = fuzz.trapmf(hbX, [-1,0,6,6.1]) 
hbPreDiabX = fuzz.trapmf(hbX, [5.9, 6, 6.4,6.5 ])
hbDiabX = fuzz.trapmf(hbX, [6.4, 6.5, 20, 21])


#Blood pressure data based from CDC
#low blood pressure can be a problem but is not the main concern here.
#https://www.cdc.gov/bloodpressure/about.htm
'''
Normal 	
systolic: less than 120 mm Hg
diastolic: less than 80 mm Hg
Elevated 	
systolic: 120–129 mm Hg
diastolic: less than 80 mm Hg
High blood pressure (hypertension) 	
systolic: 130 mm Hg or higher
diastolic: 80 mm Hg or higher
'''
#SBP
sbpX = np.arange(phrStats['SBP']['min'], phrStats['SBP']['max'])
sbpNormalX = fuzz.trapmf(sbpX, [-1,0,120, 122]) 
sbpElevateX = fuzz.trapmf(sbpX, [95, 100, 129, 140])
sbpHighX = fuzz.trapmf(sbpX, [120, 130, 160, 170])

#DBP
dbpX = np.arange(phrStats['DBP']['min'], phrStats['DBP']['max'])
dbpNormalX = fuzz.trapmf(dbpX, [-1,0,80, 85 ]) 
dbpElevateX = fuzz.trapmf(dbpX, [-1,0,80, 85 ]) #according to the experts these are thesame
dbpHighX = fuzz.trapmf(dbpX, [75, 80, 130, 131])


#LDL
'''
Less than 100mg/dL	Optimal
100-129mg/dL	Near optimal/above optimal
130-159 mg/dL	Borderline high
160-189 mg/dL	High
'''
ldlX = np.arange(phrStats['LDL']['min'], phrStats['LDL']['max'])
#https://medlineplus.gov/ldlthebadcholesterol.html
ldlOptimalX = fuzz.trapmf(ldlX, [-1,0,100, 110]) 
ldlNearOptX = fuzz.trapmf(ldlX, [95, 100, 129, 140])
ldlBordrlineX = fuzz.trapmf(ldlX, [120, 130, 160, 170])
ldlHighX = fuzz.trapmf(ldlX, [150, 160, 250, 251])


#BMI
#https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm
'''
Underweight = <18.5
Normal weight = 18.5–24.9
Overweight = 25–29.9
Obesity = BMI of 30 or greater 
'''
bmiX = np.arange(phrStats['BMI']['min'], phrStats['BMI']['max'])
bmiUnderX =  fuzz.trapmf(bmiX, [-1, 1, 18, 20 ])
bmiNormalX = fuzz.trapmf(bmiX,[18, 19, 25, 27])
bmiOverX =   fuzz.trapmf(bmiX,  [25, 27, 30, 32])
bmiObeseX =  fuzz.trapmf(bmiX, [29,30, 90, 91])

import matplotlib.pyplot as plt
# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=10, figsize=(9, 9))

ax0.plot(ageX, ageLowX, 'b', linewidth=1.5, label='Young')
ax0.plot(ageX, ageMidX, 'g', linewidth=1.5, label='Middle')
ax0.plot(ageX, ageHighX, 'r', linewidth=1.5, label='Old')
ax0.set_title('Age')
ax0.legend()

ax1.plot(incomeX, incomeLowX, 'b', linewidth=1.5, label='Poor')
ax1.plot(incomeX, incomeMidX, 'g', linewidth=1.5, label='Middle')
ax1.plot(incomeX, incomeHighX, 'r', linewidth=1.5, label='Well off')
ax1.set_title('Income')
ax1.legend()

ax2.plot(enX, enLowX, 'b', linewidth=1.5, label='Low')
#ax2.plot(enX, enMiddleX, 'g', linewidth=1.5, label='Medium')
ax2.plot(enX, enHighX, 'r', linewidth=1.5, label='High')
ax2.set_title('engagement')
ax2.legend()

ax3.plot(logX, logLowX, 'b', linewidth=1.5, label='Low')
#ax2.plot(enX, enMiddleX, 'g', linewidth=1.5, label='Medium')
ax3.plot(logX, logHighX, 'r', linewidth=1.5, label='High')
ax3.set_title('Number of Logins')
ax3.legend()

ax4.plot(hbX, hbNormalX, 'b', linewidth=1.5, label='Normal')
ax4.plot(hbX, hbPreDiabX, 'g', linewidth=1.5, label='Pre-Diabetic')
ax4.plot(hbX, hbDiabX, 'r', linewidth=1.5, label='Diabetic')
ax4.set_title('HB1AC')
ax4.legend()

ax5.plot(sbpX, sbpNormalX, 'b', linewidth=1.5, label='Normal')
ax5.plot(sbpX, sbpElevateX, 'g', linewidth=1.5, label='Elevated')
ax5.plot(sbpX, sbpHighX, 'r', linewidth=1.5, label='High')
ax5.set_title('SBP')
ax5.legend()

ax6.plot(dbpX, dbpNormalX, 'b', linewidth=1.5, label='Low')
ax6.plot(dbpX, dbpElevateX, 'g', linewidth=1.5, label='Medium')
ax6.plot(dbpX, dbpHighX, 'r', linewidth=1.5, label='High')
ax6.set_title('DBP')
ax6.legend()

ax7.plot(ldlX, ldlOptimalX , 'b', linewidth=1.5, label='Optimal')
ax7.plot(ldlX, ldlNearOptX, 'g', linewidth=1.5, label='Near Optimal')
ax7.plot(ldlX, ldlBordrlineX, 'y', linewidth=1.5, label='Borderline')
ax7.plot(ldlX, ldlHighX, 'r', linewidth=1.5, label='High')
ax7.set_title('LDL')
ax7.legend()

ax8.plot(bmiX, bmiUnderX , 'b', linewidth=1.5, label='Optimal')
ax8.plot(bmiX, bmiNormalX, 'g', linewidth=1.5, label='Near Optimal')
ax8.plot(bmiX, bmiOverX, 'y', linewidth=1.5, label='Borderline')
ax8.plot(bmiX, bmiObeseX, 'r', linewidth=1.5, label='High')
ax8.set_title('BMI')
ax8.legend()


# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

'''
start with 3
get lowest
min-1, min, 25%, median between 50 and 25%
get middle
25,median between 25 and 50, median between 50 and 75%, 75
get high
median betwen 50 and 75%, 75, max, max+1

median bewtween could also be the midpoint confidence interval for example 25-50 =37.5d

st.t.interval(alpha = 0.125, df = phrDF['Logins'].shape[0] -1, loc=np.mean(phrDF['Logins']), scale=st.sem(phrDF['Logins']))


'''
#takes a dataframe and gets the high/medium/low values in fuzzy form#
#inputs are a dataframe and a list of indices you wish to use.
def fuzzyFromStats(df):
    low = []
    med = []
    high =[]
    dfStats = phrDF.describe([.25,.375, .5, .625, .75])
    lowDict = {}
    midDict = {}
    highDict = {}
    for i in df:
        thirtySeven5 = dfStats[i]["37.5%"] 
        sixtyTwo5 = dfStats[i]["62.5%"] 
        #=  st.t.interval(alpha = 0.125, df = df[i].shape[0] -1, loc=np.mean(df[i]), scale=st.sem(df['Logins']))
        twenty5 = dfStats[i]["25%"]
        median = dfStats[i]["50%"]
        seventy5 = dfStats[i]["75%"]
        min1 = dfStats[i]["min"]
        max1 = dfStats[i]["max"]
        print(i, twenty5, thirtySeven5, sixtyTwo5, seventy5)
        
        varXAxis = np.arange(dfStats[i]['min'], dfStats[i]['max'])
        low.append(fuzz.trapmf(varXAxis, [min1-1, min1, twenty5, median ]))
        med.append(fuzz.trapmf(varXAxis, [twenty5, thirtySeven5, sixtyTwo5, seventy5]))
        high.append(fuzz.trapmf(varXAxis, [sixtyTwo5, seventy5, max1, max1+1]))
        
        lowDict[i] = [(fuzz.trapmf(varXAxis, [min1-1, min1, twenty5, median ]))]
        midDict[i] = [(fuzz.trapmf(varXAxis, [twenty5, thirtySeven5, sixtyTwo5, seventy5]))]
        highDict[i]= [(fuzz.trapmf(varXAxis, [sixtyTwo5, seventy5, max1, max1+1]))]
        
        
    return low, med, high, lowDict, midDict, highDict
        #get min/max, median, 
        #st.t.interval(alpha = 0.125, df = phrDF['Logins'].shape[0] -1, loc=np.mean(phrDF['Logins']), scale=st.sem(phrDF['Logins']))
        
"l, m, h are lists: and ld, md, and hd are dictionaries"
#automatically generated low, med, high fuzzy classifications
l, m, h, ld, md, hd = fuzzyFromStats(phrDF)
'''
#dataframes for the 
lowFuzzyMemDF = pd.DataFrame(ld)
medFuzzyMemDF = pd.DataFrame(md)
highFuzzyMemDF = pd.DataFrame(hd)

ax9.plot(incomeX, lowFuzzyMemDF['Income'] , 'b', linewidth=1.5, label='lowow')
ax9.plot(incomeX, medFuzzyMemDF['Income'], 'g', linewidth=1.5, label='Medium')
ax9.plot(incomeX, highFuzzyMemDF['Income'], 'y', linewidth=1.5, label='High')
ax9.set_title('hba1c')
ax9.legend()

'''

#use stats to determine the trapezoid functions?
#find average numbers to compare these to as well


#c-means fuzzy clustering  vs k-means 
#take out binary variables
#cluster statistics. 




#variables to test against female/caucasian/smoker/insurance?

#outlier detection, somewhat unusual, vs very unusual, vs jj
#Gowers distance/Jaccard index/hamming clustering?/ other fuzzy similarity measures?