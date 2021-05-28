import pandas as pd
import numpy as np
import quaternion as Q
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def median_filter9(d):
    res = np.zeros(d.shape, dtype='float64')
    for j in range(0, d.shape[0]):
        res[j] = d[j]
        if 3 < j < d.shape[0] - 4:
            res[j] = np.median(d[j - 4:j + 5])
    return res


def median_filter7(d):
    res = np.zeros(d.shape, dtype='float64')
    for j in range(0, d.shape[0]):
        res[j] = d[j]
        if 2 < j < d.shape[0] - 3:
            res[j] = np.median(d[j - 3:j + 4])
    return res


def median_filter5(d):
    res = np.zeros(d.shape, dtype='float64')
    for j in range(0, d.shape[0]):
        res[j] = d[j]
        if 1 < j < d.shape[0] - 2:
            res[j] = np.median(d[j - 2:j + 3])
    return res


def median_filter3(d):
    res = np.zeros(d.shape, dtype='float64')
    for j in range(0, d.shape[0]):
        res[j] = d[j]
        if 0 < j < d.shape[0] - 1:
            res[j] = np.median(d[j - 1:j + 2])
    return res


def calc_dt(d):
    res = np.zeros(d.shape[0], dtype='float64')
    for j in range(0, d.shape[0]):
        if j == 0:
            dt = d[1] - d[0]
        elif j == d.shape[0] - 1:
            dt = d[-1] - d[-2]
        else:
            dt = d[j] - d[j - 1]
        res[j] = dt
    return res


def calc_speed(angles, dt):
    res = np.zeros(angles.shape[0], dtype='float64')
    for j in range(0, angles.shape[0]):
        if j == 0:
            da = angles[1] - angles[0]
        elif j == angles.shape[0] - 1:
            da = angles[-1] - angles[-2]
        else:
            da = angles[j] - angles[j - 1]
        res[j] = da / dt[j]  # speed
    return res


def calc_time_sum(d, dt, threshold):
    total_sum = 0.0
    threshold_sum = 0.0
    for j in range(0, d.shape[0]):
        total_sum += dt[j]
        if threshold > d[j] > -threshold:
            threshold_sum += dt[j]
    return threshold_sum, total_sum, threshold_sum/total_sum


def normalize_quaternion(q):
    return q/q.norm()


def get_conjugate(q):
    return np.conjugate(q)


def quaternion_product(q1, q2):
    return q1 * q2


def quaternion_to_angular_velocity(q1, q2):
    orient_prev = Q.from_float_array(q1)
    orient_cur = Q.from_float_array(q2)
    delta_q = normalize_quaternion(quaternion_product(orient_cur, get_conjugate(orient_prev)))
    delta_q_len = np.linalg.norm(Q.as_float_array(delta_q)[1:])
    delta_q_angle = 2*np.arctan2(delta_q_len, Q.as_float_array(delta_q)[0])
    #  w = delta_q[1:] * delta_q_angle * fs
    w = Q.as_float_array(delta_q)[1:] * delta_q_angle
    return w


soldatov = "./Солдатов_Игорь_2021_05_14_14_44_36.txt"
broman = "./Броман_Лев_2021_05_14_14_49_29.txt"

data_df = pd.read_csv(broman, sep=';', skiprows=0, header=None)
raw_data = data_df.values
raw_data = raw_data.astype('float64')
raw_data[:, 0] = raw_data[:, 0] - raw_data[0, 0]  # shift to zero
raw_data[:, 0] = raw_data[:, 0] / 10000000  # to sec
times = raw_data[:, 0]

eye_id = 1
head_id = 6
data_eye_angle = raw_data[:, eye_id]

#  Преобразуем кватернионы в углы Крилова
head_angles = np.zeros((raw_data.shape[0], 3), dtype='float64')
for i in range(0, raw_data.shape[0]-1):
    q = R.from_quat(raw_data[i, 3:7])
    head_angles[i, :] = q.as_euler('zyx', degrees=True)

plt.plot(times, head_angles[:, 0], label='X')
plt.plot(times, head_angles[:, 1], label='Y')
plt.plot(times, head_angles[:, 2], label='Z')
plt.legend(['X', 'Y', 'Z'])
plt.xlabel('Время, сек')
plt.ylabel('Угол, град')
plt.show()

head_angles_y = head_angles[:, 1]

plt.plot(times[:-1], head_angles_y[:-1], label='Y')
plt.legend(['Y'])
plt.xlabel('Время, сек')
plt.ylabel('Угол, град')
plt.show()

dt = calc_dt(raw_data[:, 0])
eye_speed = -calc_speed(data_eye_angle, dt)
eye_speed_filter = median_filter5(eye_speed)
head_speed = calc_speed(head_angles_y, dt)
head_speed[-1] = 0
head_speed_filter = median_filter5(head_speed)
total_speed = head_speed - eye_speed
total_speed_filter = head_speed_filter - eye_speed_filter

plt.plot(times, head_speed, label='Скорость головы')
plt.plot(times, head_speed_filter, label='Скорость головы, медианный фильтр 5')
plt.plot(times, np.zeros(times.shape), label='Ось X')
plt.legend(['Скорость головы', 'Скорость головы, медианный фильтр 5', 'Ось X'])
plt.xlabel('Время, сек')
plt.ylabel('Угловая скорость, град/сек')
plt.show()

#  ГЛАЗ
plt.plot(times, data_eye_angle, label='Угол поворота глаза')
plt.legend(['Угол поворота глаза'])
plt.xlabel('Время, сек')
plt.ylabel('Угол, град')
plt.show()

plt.plot(times, eye_speed, label='Скорость глаза')
plt.plot(times, eye_speed_filter, label='Скорость глаза, медианный фильтр 5')
plt.plot(times, np.zeros(times.shape), label='Ось X')
plt.legend(['Скорость глаза', 'Скорость глаза, медианный фильтр 5', 'Ось X'])
plt.xlabel('Время, сек')
plt.ylabel('Угловая скорость, град/сек')
plt.show()

plt.plot(times, eye_speed_filter, label='Скорость глаза, медианный фильтр 5')
plt.plot(times, np.zeros(times.shape), label='Ось X')
plt.legend(['Скорость глаза, медианный фильтр 5', 'Ось X'])
plt.xlabel('Время, сек')
plt.ylabel('Угловая скорость, град/сек')
plt.show()

#  ГОЛОВА И ГЛАЗ
plt.plot(times, total_speed_filter, label='Скорость глаза + скорость головы')
plt.plot(times, head_speed_filter, label='Скорость головы, медианный фильтр 5')
plt.plot(times, eye_speed_filter, label='Скорость глаза, медианный фильтр 5')
plt.plot(times, np.zeros(times.shape), label='Ось X')
plt.legend(['Скорость глаза + скорость головы', 'Скорость головы, медианный фильтр 5', 'Скорость глаза, медианный фильтр 5', 'Ось X'])
plt.xlabel('Время, сек')
plt.ylabel('Угловая скорость, град/сек')
plt.show()

data_up = np.zeros(total_speed_filter.shape, dtype='float64')
data_up_tmp = total_speed_filter.copy()
data_up_tmp[data_up_tmp < 4] = 0
data_up = data_up + data_up_tmp
data_up_tmp = total_speed_filter.copy()
data_up_tmp[data_up_tmp > -4] = 0
data_up = data_up + data_up_tmp

data_down = total_speed_filter.copy()
data_down[data_down > 4] = 0
data_down[data_down < -4] = 0

plt.plot(times, data_up, label='Модуль скорости больше порога')
plt.plot(times, data_down, label='Модуль скорости меньше порога')
plt.plot(times, np.zeros(times.shape), label='Ось X')
plt.legend(['Модуль скорости больше порога', 'Модуль скорости меньше порога', 'Ось X'])
plt.xlabel('Время, сек')
plt.ylabel('Угловая скорость, град/сек')
plt.show()


total_speed_filter_1 = [i for i, j in zip(total_speed_filter, times) if 67 < j < 78]
dt_1 = [i for i, j in zip(dt, times) if 24 < j < 36]

for thresh in range(4, 18):
    thresh_sum, tot_sum, coeff = calc_time_sum(np.array(total_speed_filter_1), np.array(dt_1), thresh)
    print(f"Threshold: {thresh}, threshold sum: {thresh_sum}, total sum: {tot_sum}, gaze coefficient: {coeff}")

for thresh in range(4, 18):
    thresh_sum, tot_sum, coeff = calc_time_sum(total_speed_filter, dt, thresh)
    print(f"Threshold: {thresh}, threshold sum: {thresh_sum}, total sum: {tot_sum}, gaze coefficient: {coeff}")