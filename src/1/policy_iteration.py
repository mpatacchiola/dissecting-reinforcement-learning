import numpy as np

def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1,12))
            v[0,s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return u

def return_policy_evaluation_linalg(p, r, T, gamma):
    u = np.zeros(12)
    for s in range(12):
        if not np.isnan(p[s]):
            action = int(p[s])
            u[s] = np.linalg.solve(np.identity(12) - gamma*T[:,:,action], r)[s]
    return u

def return_expected_action(p, u, T, v):
    actions_array = np.zeros(4)
    for action in range(4):
         #Expected utility of doing a in state s, according to T and u.
         actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)

def print_policy(p, shape):
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "           
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def main_iterative():
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")

    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        u1 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        #Stopping criteria
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma: break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(p, u, T, v)         
                if a != p[s]: p[s] = a
        print_policy(p, shape=(3,4))

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")


def main_linalg():
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")

    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1

    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])

    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])

    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        #u1 = u.copy()
        u = return_policy_evaluation_linalg(p, r, T, gamma)
        #Stopping criteria
        #delta = np.absolute(u - u1).max()
        #if (delta < epsilon * (1 - gamma) / gamma) or iteration > 100: break
        unchanged = True
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(p, u, T, v)         
                if a != p[s]: 
                    p[s] = a
                    unchanged = False
        print_policy(p, shape=(3,4))

        if unchanged: break

    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    #print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")




def main():

    main_iterative()
    #main_linalg()


if __name__ == "__main__":
    main()
