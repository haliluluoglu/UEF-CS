##Exercise1
#Question 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path= "C:/Users/Halil/Desktop/Pattern/Exercises/1.exercise/bogus_student_data.txt"

data=pd.read_csv(path, sep=' ')

final_grade = data['exercise_points']
groups = data.groupby(by='grades')
exercise_points=groups['exercise_points']
for grade, points in exercise_points:
    print("Grades: %d" %grade, np.mean(points), np.std(points))

plt.figure(figsize=(10,5))
for i,(grade,points) in enumerate(exercise_points):
    plt.subplot(1,6,i+1)
    plt.hist(points,bins=25)
    plt.title("Grades: %d" % grade)
plt.show()
