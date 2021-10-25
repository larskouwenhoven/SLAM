# Jinwook Huh
import numpy as np
import pickle
from scipy import io
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import math
import MapUtils_fclad as mu

def get_lidar(file_name):
    data = io.loadmat(file_name+".mat")
    lidar = []
    angles = np.double(data['Hokuyo0']['angles'][0][0])
    ranges = np.array(data['Hokuyo0']['ranges'][0][0]).T
    ts_set = data['Hokuyo0']['ts'][0,0][0]
    idx = 0	
    for m in ranges:
        tmp = {}
        tmp['t'] = ts_set[idx]
        tmp['scan'] = m
        tmp['angle'] = angles
        lidar.append(tmp)
        idx = idx + 1
    return lidar


def get_encoder(file_name):
    data = io.loadmat(file_name+".mat")
    #	pdb.set_trace()
    FR = np.double(data['Encoders']['counts'][0,0][0])
    FL = np.double(data['Encoders']['counts'][0,0][1])
    RR = np.double(data['Encoders']['counts'][0,0][2])
    RL = np.double(data['Encoders']['counts'][0,0][3])
    ts = np.double(data['Encoders']['ts'][0,0][0])
    return FR, FL, RR, RL, ts	


def get_imu(file_name):
    data = io.loadmat(file_name+".mat")
    acc_x = np.double(data['vals'])[0]
    acc_y = np.double(data['vals'])[1]
    acc_z = np.double(data['vals'])[2]
    gyro_x = np.double(data['vals'])[3]
    gyro_y = np.double(data['vals'])[4]
    gyro_z = np.double(data['vals'])[5]	
    ts = np.double(data['ts'][0])
    #	pdb.set_trace()
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ts


def read_lidar(angles, ranges, current_angle):
    x_wall = []
    y_wall = []
    data = np.zeros((1000, 1000))
    x_offset = 250
    y_offset = 250
    min_lidar_range = (584.2 - 330.2) / 2 / 10 + 150
    for i in range(0, len(angles)):
        current_range = ranges[i] * 10
        if (ranges[i] * 100 > min_lidar_range) & (ranges[i] < 20):
            if current_angle > np.pi:
                current_angle -= 2. * np.pi
            if current_angle < -np.pi:
                current_angle += 2. * np.pi
            y = current_range * np.cos(angles[i] + current_angle)
            x = current_range * np.sin(angles[i] + current_angle)
            data[int(x) + x_offset, int(y) + y_offset] = 1
            x_wall.append(x)
            y_wall.append(y)
    return data, x_wall, y_wall


particles = []
angleVar = 5 * 10**-4

def advance_robot(FR_t, FL_t, particles, noise):
    width = (476.25 - (476.25-311.15)/2) - 35 
    wheel_radius = (584.2-330.2)/2 
    wheel_circumference = 2*wheel_radius*math.pi

    dR = FR_t / 360 * wheel_circumference
    dL = FL_t / 360 * wheel_circumference

    for i in range(len(particles)):
        x0, y0, angle0, weight = particles[i]
        angleChange = ((dR - dL) / width) #+ np.random.randn() * noise

        angle = angleChange / 2 + angle0 

        dY = (dL + dR)/2*math.cos(angle) + np.random.randn() * noise
        dX = (dL + dR)/2*math.sin(angle) + np.random.randn() * noise

        particles[i] = [x0 + dX / 100, y0 + dY / 100, angle + np.random.randn() * noise, weight]
    return particles


def get_correlation(map, xs0, ys0, current_x, current_y):
    lidar_map = np.zeros((601, 601))
    for i in range(len(xs0[0])):
        x = int(xs0[0][i] + current_x)
        y = int(ys0[0][i] + current_y)
        lidar_map[y, x] = 1
    return np.sum(lidar_map * map)


def update_particle_weights(particles, lidar_t, map):
    ranges = lidar_t['scan']
    angles = lidar_t['angle'].reshape(ranges.shape[0],)
    indValid = np.logical_and((ranges < 10),(ranges> 0.15))
    ranges = ranges[indValid]
    angles = angles[indValid]
    xs = ys = np.arange(-4,5,1)
    max_cor = []
    ys0 = np.array([ranges*np.cos(angles)])
    xs0 = np.array([ranges*np.sin(angles)])
    x_im = np.arange(0, 601, 1) #x-positions of each pixel of the map
    y_im = np.arange(0, 601, 1) #y-positions of each pixel of the map
    particles = np.array(particles)
    for i in range(len(particles)):
        particle = particles[i]
        x_range = (particle[0]+xs)
        y_range = (particle[1]+ys)
        simple_cor = get_correlation(map, xs0, ys0, particle[0], particle[1])
        max_cor.append(simple_cor)
    max_cor = particles[:,3] * max_cor
    e = np.exp(max_cor-np.max(max_cor))
    particles[:,3] = e / e.sum()
    return particles


def SLAM(lidar, FR, FL, enc_ts, n_particles=50, noise=angleVar):
    path_map = np.zeros((600, 600))
    map = np.zeros((601, 601))
    angle = 0
    x_start = 200
    y_start = 200
    particle_weight = 1 / n_particles
    particles = [[x_start, y_start, angle, particle_weight] for i in range(n_particles)]
    for t in range(0, len(enc_ts)-100):
        if t % 100 == 0:
            print(t)
        # 1. Advance robot based on odometry (all particles)
        particles = advance_robot(FR[t], FL[t], particles, noise)
        particles = np.array(particles)
        # plot first particle as a sanity check
        max_weight = np.argmax(particles[:,3])
        #print(max_weight)
        rx, ry, _, _ = particles[max_weight]
        path_map[int(rx), int(ry)] = 1
        
        # 2. Update the particle weights based on Lidar data
        particles = update_particle_weights(particles, lidar[t], map)

        # 3. Resample if necessary
        n_effective = np.sum(particles[:,3])**2 / np.sum(particles[:,3]**2)
        new_particles = []
        if n_effective < n_particles / 2:
            cumsum = np.cumsum(particles[:,3])
            for i in range(n_particles):
                random_number = np.random.random()
                index = np.argmax(cumsum > random_number)
                new_particles.append(particles[index])
            particles = np.array(new_particles)
            particles[:,3] = 1 / n_particles
            #print("New particles", particles)
            
        # 4. Update the map
        max_weight = np.argmax(particles[:,3])
        best_particle = particles[max_weight]
        x_start, y_start, angle, weight = best_particle
        # Get the ray data
        _, x_wall, y_wall = read_lidar(lidar[t]['angle'], lidar[t]['scan'], angle)
        for i in range(len(x_wall)):
            x_to_update = x_wall[i] + x_start
            y_to_update = y_wall[i] + y_start
            if map[int(x_to_update), int(y_to_update)] < 20:
                map[int(x_to_update), int(y_to_update)] += 0.9
        #print(np.asarray(x_wall).shape, np.asarray(y_wall).shape)
        free = mu.getMapCellsFromRay_fclad(
            x_start, 
            y_start, 
            np.array(x_wall).reshape(len(x_wall),).astype(np.int16), 
            np.array(y_wall).reshape(len(y_wall),).astype(np.int16), 
            600
        )
        free_y, free_x = free[0], free[1]
        for i in range(len(free_x)):
            if map[free_y[i], free_x[i]] > -20:
                map[free_y[i], free_x[i]] -= 0.1 
        n_effective = np.sum(particles[:,3])**2 / np.sum(particles[:,3]**2)
    return path_map, map, particles


for i in ["20", "21", "22", "23", "24"]:
    print("Working on map", i)
    encoders = get_encoder("data/Encoders" + i)
    lidar = get_lidar("data/Hokuyo" + i)
    # Create encoder df
    enc_df = pd.DataFrame(encoders).T
    enc_df.rename(columns = {0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL', 4: 'ts'}, inplace=True)
    path_map, map, particles = SLAM(lidar, enc_df.FR.values, enc_df.FL.values, enc_df.ts.values)
    new_map = np.where((map < 20) & (map > -20), 0, map)
    plt.imsave('path' + i + '.png', path_map)
    plt.imsave('map' + i + '.png', new_map, cmap='Greys')
    
