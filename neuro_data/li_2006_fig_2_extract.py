# -------------------------------------------------------------------------------------------------
#  Extracted Data from Figure C3 and C4 of
#  [Li, Piech and Gilbert - 2006 - Contour Saliency in primary Visual Cortex]
#  is stored in Li2006.pickle
#
#  Data is extracted from figures using http://arohatgi.info/WebPlotDigitizer/
#
# Author: Salman Khan
# Date  : 30/08/17
# -------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pickle

contour_len_ma_len = np.array([1, 3, 5, 7, 9])
contour_len_ma_gain = np.array([1, 1.7, 2.14, 2.41, 2.71])

contour_len_mb_len = np.array([1, 3, 5, 7, 9])
contour_len_mb_gain = np.array([1, 1.4, 1.85, 2.10, 2.27])

contour_separation_ma_rcd = np.array([1, 1.2, 1.4, 1.6, 1.9])
contour_separation_ma_gain = np.array([2.24, 2.20, 1.73, 1.24, 1])

contour_separation_mb_rcd = np.array([1, 1.2, 1.4, 1.6, 1.9])
contour_separation_mb_gain = np.array([1.96, 1.69, 1.28, 1.16, 1.16])

contour_len_avg_len = np.array([1, 3, 5, 7, 9])
contour_len_avg_gain = (contour_len_ma_gain + contour_len_mb_gain) / 2.0

contour_separation_avg_rcd = np.array([1, 1.2, 1.4, 1.6, 1.9])
contour_separation_avg_gain = (contour_separation_ma_gain + contour_separation_mb_gain) / 2.0


plt.figure()
plt.plot(contour_len_ma_len, contour_len_ma_gain, label='ma')
plt.plot(contour_len_mb_len, contour_len_mb_gain, label='mb')
plt.plot(contour_len_avg_len, contour_len_avg_gain, label='avg')
plt.legend()
plt.xlabel('contour length')
plt.ylabel('Average Gain')

plt.figure()
plt.plot(contour_separation_ma_rcd, contour_separation_ma_gain, label='ma')
plt.plot(contour_separation_mb_rcd, contour_separation_mb_gain, label='mb')
plt.plot(contour_separation_avg_rcd, contour_separation_avg_gain, label='avg')
plt.legend()
plt.xlabel("relative colinear distance")
plt.ylabel("Average Gain")
plt.legend()


results_dict = {
    'contour_len_ma_len': contour_len_ma_len, 'contour_len_ma_gain': contour_len_ma_gain,
    'contour_len_mb_len': contour_len_mb_len, 'contour_len_mb_gain': contour_len_mb_gain,
    'contour_separation_ma_rcd': contour_separation_ma_rcd,
    'contour_separation_ma_gain': contour_separation_ma_gain,
    'contour_separation_mb_rcd': contour_separation_mb_rcd,
    'contour_separation_mb_gain': contour_separation_mb_gain, 'contour_len_avg_len': contour_len_avg_len,
    'contour_len_avg_gain': contour_len_avg_gain, 'contour_separation_avg_rcd': contour_separation_avg_rcd,
    'contour_separation_avg_gain': contour_separation_avg_gain
}

with open('Li2006.pickle', 'wb') as handle:
    pickle.dump(results_dict, handle)


# now to test to see if data was successfully stored

with open('Li2006.pickle', 'rb') as handle:
    new_data = pickle.load(handle)

plt.figure()
plt.plot(new_data['contour_len_avg_len'], new_data['contour_len_avg_gain'])
plt.title("restored data")
plt.figure()
plt.plot(new_data['contour_separation_avg_rcd'], new_data['contour_separation_avg_gain'])






