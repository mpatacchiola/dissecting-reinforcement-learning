import numpy as np

#Trowing a dice for N times and evaluating the expectation
dice = np.random.randint(low=1, high=7, size=10)
print("Expectation (10 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100)
print("Expectation (100 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=1000)
print("Expectation (1000 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100000)
print("Expectation (100000 times): " + str(np.mean(dice)))
