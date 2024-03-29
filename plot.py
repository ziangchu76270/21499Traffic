import numpy as np
import matplotlib.pyplot as plt

"""
plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          [26.666666666666668, 
          106.66666666666667, 
          240.0, 
          426.6666666666667,
          666.6666666666667, 
          960.0, 
          1306.6666666666665, 
          1706.6666666666667, 
          2160.0, 
          2666.666666666667], 'ro',
          [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          [26.666666666666668, 
          106.66666666666667, 
          240.0, 
          426.6666666666667,
          666.6666666666667, 
          960.0, 
          1306.6666666666665, 
          1706.6666666666667, 
          2160.0, 
          2666.666666666667],'r')
plt.plot([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          [9.35377137947473, 
          25.185570368894368, 
          54.75128199350902, 
          96.98712577661168, 
          150.73341487731648, 
          218.15547257246047, 
          291.11943531476254, 
          383.0649658057219, 
          476.024331959737, 
          588.8067027652917], 'bo',
          [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
          [9.35377137947473, 
          25.185570368894368, 
          54.75128199350902, 
          96.98712577661168, 
          150.73341487731648, 
          218.15547257246047, 
          291.11943531476254, 
          383.0649658057219, 
          476.024331959737, 
          588.8067027652917],'b')
plt.ylabel('cost')
plt.xlabel('number of vehicles')
plt.show()
"""

xr1 = np.arange(5, 51, 5)
plt.plot(xr1,
          [6.666666666666667, #naive
          26.666666666666668,
          60.0,                 #15 
          106.66666666666667, 
          166.66666666666669, #25
          240.0, 
          326.66666666666663,
          426.6666666666667,
          540.0,
          666.6666666666667], 'ro',
          xr1,
          [6.666666666666667,
          26.666666666666668, 
          60.0,                     #15
          106.66666666666667, 
          166.66666666666669, #25
          240.0, 
          326.66666666666663,
          426.6666666666667,
          540.0,
          666.6666666666667],'r')
plt.plot(xr1,
          [4.073732323988, #optimized
          9.35377137947473,
          15.729517533812476, #15 
          25.185570368894368, 
          37.1249019370484, #25
          54.75128199350902,
          74.29059171244955,
          96.98712577661168, 
          121.87916900800897,
          150.73341487731648], 'bo',
          xr1,
          [4.073732323988,
          9.35377137947473,
          15.729517533812476, #15 
          25.185570368894368,
          37.1249019370484,
          54.75128199350902,
          74.29059171244955,
          96.98712577661168,
          121.87916900800897,
          150.73341487731648],'b')
plt.ylabel('Cost', fontname='Times New Roman', fontsize=14)
plt.xlabel('Number of Vehicles', fontname='Times New Roman', fontsize=14)
plt.gca().yaxis.set_label_coords(-0.1,.5)
plt.show()

"""
Incomplete 5
    x = 5:
        naive cost: 6.666666666666667
        optimized cost: 4.073732323988
    
    x = 10:
        original cost:  26.666666666666668
        optimized cost:  9.35377137947473
    
    x = 15:
        naive: 60.0
        optim: 15.729517533812476

    x = 20:
        original cost:  106.66666666666667
        optimized cost:  25.185570368894368

    x = 25:
        naive: 166.66666666666669
        optim: 37.1249019370484

    x = 30:
        original cost: 240.0
        optimized cost: 54.75128199350902

    x = 35:
        naive:326.66666666666663
        optim: 74.29059171244955

    x = 40:
        original cost: 426.6666666666667
        optimized cost: 96.98712577661168

    x = 45:
        naive:
        optim:

    x = 50:
        original cost:  666.6666666666667
        optimized cost: 150.73341487731648

    x = 60:
        original cost:  960.0
        optimized cost: 218.15547257246047

    x = 70:
        original cost: 1306.6666666666665
        optimized cost: 291.11943531476254
  
    x = 80:
        original cost: 1706.6666666666667
        optimized cost: 383.0649658057219

    x = 90:
        original cost: 2160.0
        optimized cost: 476.024331959737
  
    x = 100:
        original cost: 2666.666666666667
        optimized cost: 588.8067027652917

"""
"""
Default:
    original cost:  241.62455869416718
    optimized cost: 108.97609925935762

"""

"""
CHANGE NUMBER OF STEPS: (DEFAULT)
original: 241.62455869416718
2 steps: 108.97609925935762
3 steps: 97.62373048593889
4 steps: 93.4057109504054
5 steps: 89.82794463228667
6 steps: 87.60402961350468
7 steps: 87.46472449831589
8 steps: 86.92922541587792
9 steps: 86.08701094029519
10 steps:86.55939482675895
"""

"""
plt.plot([2, 3, 4, 5, 6, 7, 8, 9],
        [108.97609925935762, 
         97.62373048593889, 
         93.4057109504054,
         89.82794463228667,
         87.60402961350468,
         87.46472449831589,
         86.92922541587792,
         86.08701094029519])
plt.show()
"""
"""
datarange = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
optimized = [9.35377137947473, 
          25.185570368894368, 
          54.75128199350902, 
          96.98712577661168, 
          150.73341487731648, 
          218.15547257246047, 
          291.11943531476254, 
          383.0649658057219, 
          476.024331959737, 
          588.8067027652917]
naive = [26.666666666666668, 
          106.66666666666667, 
          240.0, 
          426.6666666666667,
          666.6666666666667, 
          960.0, 
          1306.6666666666665, 
          1706.6666666666667, 
          2160.0, 
          2666.666666666667]
"""
def f(x, y):
    result = []
    for i in range (len(x)):
        result.append(x[i] / y[i])
    return result
"""
# percent 
plt.plot(datarange, f(optimized, naive))
axes = plt.gca()
axes.set_xlim([10,100])
axes.set_ylim([0,0.5])
# plt.show()
"""
optimized1 = [4.073732323988,
          9.35377137947473,
          15.729517533812476, #15 
          25.185570368894368,
          37.1249019370484,
          54.75128199350902,
          74.29059171244955,
          96.98712577661168,
          121.87916900800897,
          150.73341487731648]
naive1 = [6.666666666666667, #naive
          26.666666666666668,
          60.0,                 #15 
          106.66666666666667, 
          166.66666666666669, #25
          240.0, 
          326.66666666666663,
          426.6666666666667,
          540.0,
          666.6666666666667]
plt.plot(xr1, f(optimized1, naive1), 'ro',
         xr1, f(optimized1, naive1), 'r')
axes = plt.gca()
axes.set_xlim([10,50])
axes.set_ylim([0.22,0.38])
plt.ylabel('Solution Quality', fontname='Times New Roman', fontsize=14)
plt.xlabel('Number of vehicles',fontname='Times New Roman', fontsize=14)
axes.yaxis.set_label_coords(-0.1,.5)
#Comic Sans MS
plt.show()
