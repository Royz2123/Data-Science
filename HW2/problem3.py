import numpy as np

import util

POINTS_PER_PLOT = 300
PLOTS = 2
TOTAL_POINTS = POINTS_PER_PLOT * PLOTS

def question_a():
    # generate the distribution
    x_data = np.random.uniform(-1, 1, (TOTAL_POINTS, 1))
    y_data = np.random.uniform(0, 5, (TOTAL_POINTS, 1))
    merged = np.concatenate((x_data, y_data), axis=1)
    data1, data2 = merged[:POINTS_PER_PLOT], merged[POINTS_PER_PLOT:]

    # PLOT the outputs
    for data in (data1, data2):
        util.plot_in_R2(data.T[0], data.T[1], "Uniform Distribution")

    # save the points
    util.write_list(data1, "problem3_outputs/question_a/data1.txt")
    util.write_list(data2, "problem3_outputs/question_a/data2.txt")
     

def question_b():
    # generate the distribution
    x_data = np.random.normal(1, 2, (TOTAL_POINTS, 1))
    y_data = np.random.normal(1, 2, (TOTAL_POINTS, 1))
    merged = np.concatenate((x_data, y_data), axis=1)
    data1, data2 = merged[:POINTS_PER_PLOT], merged[POINTS_PER_PLOT:]

    # PLOT the outputs
    for data in (data1, data2):
        util.plot_in_R2(data.T[0], data.T[1], "Gaussian Distribution")

    # save the points
    util.write_list(data1, "problem3_outputs/question_b/data1.txt")
    util.write_list(data2, "problem3_outputs/question_b/data2.txt")


def question_c():
    # generate the distribution
    total_data = []
    for plot_index in range(PLOTS):
        current_data = []
        # gaussian indexes to sample from
        indexes = np.random.randint(1, 4, POINTS_PER_PLOT)
        unique, number_of_samples = np.unique(indexes, return_counts=True)

        # add each gaussian to data
        x_data = np.array([])
        y_data = np.array([])
        for i in range(1, 4):
            shape = int(number_of_samples[i - 1])

            x_data = np.concatenate(
                (
                    np.random.normal(i, 2 * i, shape),
                    x_data
                ), axis = 0
            )
            y_data = np.concatenate(
                (
                    np.random.normal(i, 2 * i, shape),
                    y_data
                ), axis = 0
            )            
        total_data.append(np.dstack((np.array([x_data]), np.array([y_data]))))
            
    data1, data2 = total_data  
           
    # PLOT the outputs
    for data in (data1, data2):
        util.plot_in_R2(data.T[0], data.T[1], "Gaussian Distribution")

    # save the points
    util.write_list(data1, "problem3_outputs/question_c/data1.txt")
    util.write_list(data2, "problem3_outputs/question_c/data2.txt")    
        
        
def question_d():
    # In order to generate a circle, we will sample two parameters
    # R and theta, and from them
    obj_pnts = int(TOTAL_POINTS / 2)
    plt_obj_pnts = int(POINTS_PER_PLOT / 2)
    
    # Sample the inner circle
    circle_r_data = np.random.uniform(0, 0.5, (obj_pnts, 1))
    circle_t_data = np.random.uniform(0, 2 * np.pi, (obj_pnts, 1))
    
    # Sample the outer ring
    ring_r_data = np.random.uniform(1.5, 1.75, (obj_pnts, 1))
    ring_t_data = np.random.uniform(0, 2 * np.pi, (obj_pnts, 1))
    
    # concatenate each plot
    r_data1 = np.concatenate((circle_r_data[:plt_obj_pnts], ring_r_data[:plt_obj_pnts]), axis=0)  
    r_data2 = np.concatenate((circle_r_data[plt_obj_pnts:], ring_r_data[plt_obj_pnts:]), axis=0)  
    t_data1 = np.concatenate((circle_t_data[:plt_obj_pnts], ring_t_data[:plt_obj_pnts]), axis=0) 
    t_data2 = np.concatenate((circle_t_data[plt_obj_pnts:], ring_t_data[plt_obj_pnts:]), axis=0) 
    polar_data1 = np.concatenate((r_data1, t_data1), axis=1)
    polar_data2 = np.concatenate((r_data2, t_data2), axis=1)
        
    # turn from polar to cartesian
    total_data = []
    for data in (polar_data1, polar_data2):
        total_data.append([util.pol2cart(point[0], point[1]) for point in data])
    data1, data2 = np.array(total_data[0]), np.array(total_data[1]) 
    
    # PLOT the outputs
    for data in (data1, data2):
        util.plot_in_R2(data.T[0], data.T[1], "Circle inside a Ring")

    # save the points
    util.write_list(data1, "problem3_outputs/question_d/data1.txt")
    util.write_list(data2, "problem3_outputs/question_d/data2.txt")
        
        
question_d()