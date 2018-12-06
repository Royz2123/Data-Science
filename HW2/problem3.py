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
        
question_c()