# -------------------------------------------------------------------------------------------------
# Example Code for plotting figures, that are good for publishing in papers.
# Consistent way of increasing font sizes.
#
# Author: Salman Khan
# Date  : 05/04/18
# -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

rc = {'font.size': 45, 'axes.labelsize': 45, 'legend.fontsize': 45.0,
      'axes.titlesize': 45, 'xtick.labelsize': 45, 'ytick.labelsize': 45,
      "xtick.major.width": 5, "ytick.major.width": 5, 'axes.linewidth': 3}

plt.rcParams.update(**rc)

# -----------------------------------------------------------------------------------------------
# Figure - 1 Fragment Spacing for diagonal contour (kernel Index 54)
# -----------------------------------------------------------------------------------------------
x1 = np.array([
    1,
    1.2008185539,
    1.4005457026,
    1.6002728513,
    1.9004092769,
])

y1 = np.array([
    1.9634236374,
    1.6903532438,
    1.2796310724,
    1.1621757339,
    1.1580605104,

])

x2 = np.array([
    1,
    1.1811732606,
    1.3623465211,
    1.5467939973,
    1.7268758527,
    1.8174624829,

])

y2 = np.array([
    2.2357428394,
    1.9567608945,
    1.5161829396,
    1.2222261536,
    1.0779116329,
    1.1493932291,

])

x3 = np.array([
    1,
    1.1997271487,
    1.3994542974,
    1.5991814461,
    1.8982264666,

])

y3 = np.array([
    2.2387353581,
    2.1990855059,
    1.7285129604,
    1.2399853028,
    0.9994651842,

])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x1, y1, label='monkey b', marker='s', markersize='15', linewidth=5, linestyle='--', color='green')
plt.plot(x2, y2, label='model', marker='o', markersize='15', linewidth=5, color='blue')
plt.plot(x3, y3, label='monkey a', marker='d', markersize='15', linewidth=5, linestyle='--', color='red')

plt.xlabel("Fragment Spacing")
plt.ylabel("Gain")

y_ticks = np.arange(1, 2.3, 0.4)
plt.yticks(y_ticks)

plt.legend()

# -----------------------------------------------------------------------------------------------
# Figure - 2 Contour Length for diagonal contour (54)
# -----------------------------------------------------------------------------------------------
x1 = np.array([
    0.991462877,
    3.0013879372,
    5.0030202909,
    7.0057001445,
    9.0007856248,
])

y1 = np.array([
    1.0097613883,
    1.409978308,
    1.8557483731,
    2.1062906725,
    2.2722342733,
])

x2 = np.array([
    0.9982716254,
    2.9996595626,
    4.9866269199,
    6.9893766067,
    9.0007332498,
])

y2 = np.array([
    1.2407809111,
    1.7321041215,
    1.9110629067,
    2.1485900217,
    2.2819956616,
])

x3 = np.array([
    0.9995111668,
    2.9997992292,
    5.0014664996,
    6.9960107718,
    8.9823147126,
])

y3 = np.array([
    1.0097613883,
    1.7060737527,
    2.1453362256,
    2.4121475054,
    2.7147505423,
])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x1, y1, label='monkey b', marker='s', markersize='15', linewidth=5, linestyle='--', color='green')
plt.plot(x2, y2, label='model', marker='o', markersize='15', linewidth=5, color='blue')
plt.plot(x3, y3, label='monkey a', marker='d', markersize='15', linewidth=5, linestyle='--', color='red')

plt.xlabel("Contour Length")
plt.ylabel("Gain")

y_ticks = np.arange(1, 3, 1)
x_ticks = np.arange(1, 10, 2)
plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.legend()

# -----------------------------------------------------------------------------------------------
# Figure - 3 Fragment Spacing for horizontal contour (kernel Index 5)
# -----------------------------------------------------------------------------------------------
x1 = np.array([
    1.0027778431,
    1.2022548038,
    1.4008079061,
    1.6004703906,
    1.899813066,
])

y1 = np.array([
    1.9608040201,
    1.6914572864,
    1.2814070352,
    1.1608040201,
    1.1587939698,
])

x2 = np.array([
    1.0023679357,
    1.1832962756,
    1.3652449964,
    1.546005362,
    1.7262856221,
    1.8189522605,
])

y2 = np.array([
    2.232160804,
    2.0914572864,
    1.5688442211,
    1.2934673367,
    1.2331658291,
    1.1286432161,
])

x3 = np.array([
    1.0023804711,
    1.2028865876,
    1.4006161146,
    1.6005681667,
    1.8996175138,
])

y3 = np.array([
    2.2422110553,
    2.1979899497,
    1.727638191,
    1.2391959799,
    1.0020100503,
])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x1, y1, label='monkey b', marker='s', markersize='15', linewidth=5, linestyle='--', color='green')
plt.plot(x2, y2, label='model', marker='o', markersize='15', linewidth=5, color='blue')
plt.plot(x3, y3, label='monkey a', marker='d', markersize='15', linewidth=5, linestyle='--', color='red')

plt.xlabel("Fragment Spacing")
plt.ylabel("Gain")

y_ticks = np.arange(1, 2.3, 0.4)
plt.yticks(y_ticks)

plt.legend()

# -----------------------------------------------------------------------------------------------
# Figure - 4 Contour Length for horizontal contour (5)
# -----------------------------------------------------------------------------------------------
x1 = np.array([
    1.0067453626,
    3.0033726813,
    5.0134907251,
    6.9966273187,
    9.0067453626,
])

y1 = np.array([
    1.0025961867,
    1.4010712569,
    1.8514450578,
    2.1041169493,
    2.2727774058,
])

x2 = np.array([
    1,
    3.0101180438,
    4.9932546374,
    7.0033726813,
    9,
])

y2 = np.array([
    1.1088543996,
    1.5641705399,
    2.0145360063,
    2.2227351778,
    2.3765644491,
])

x3 = np.array([
    1.0134907251,
    2.9966273187,
    5,
    7.0033726813,
    9,
])

y3 = np.array([
    1.0001271006,
    1.7025518751,
    2.1380965743,
    2.4105440741,
    2.7126435265,
])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(x1, y1, label='monkey b', marker='s', markersize='15', linewidth=5, linestyle='--', color='green')
plt.plot(x2, y2, label='model', marker='o', markersize='15', linewidth=5, color='blue')
plt.plot(x3, y3, label='monkey a', marker='d', markersize='15', linewidth=5, linestyle='--', color='red')

plt.xlabel("Contour Length")
plt.ylabel("Gain")

y_ticks = np.arange(1, 3, 1)
x_ticks = np.arange(1, 10, 2)
plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.legend()
