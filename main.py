import numpy
import random
import math
import numpy as np
from time import sleep

# ------------- Variables -------------

Pd = 7500  # POWER DEMAND

a = [0] * 10  # a factor
b = [0] * 10  # b factor
c = [0] * 10  # c factor
Pmin = [0] * 10
Pmax = [0] * 10
E = [0] * 10

# OLD Variables
# P = numpy.zeros(shape=(10, 10))
# K = [0] * 10
# K = numpy.zeros(shape=(10, 10))
# P_Ch = numpy.zeros(shape=(10, 10))
# K_Ch = numpy.zeros(shape=(10, 10))

Pconnected = numpy.zeros(shape=(20, 10))
Kconnected = numpy.zeros(shape=(20, 10))
Econnected = [0] * 20
KconnectedIndividual = [0] * 20

N = numpy.random.randint(low=1, high=32, size=(10, 10), dtype=int)
Epsilon = 100  # max loss

# print("power Level Matrix \n", N)

for x in range(10):
    a[x] = random.uniform(2.5, 7.5)
    b[x] = random.uniform(2.5 * 10 ** -2, 7.5 * 10 ** -2)
    c[x] = random.uniform(2.5 * 10 ** -4, 7.5 * 10 ** -4)
    Pmin[x] = random.randrange(250, 300)
    Pmax[x] = random.randrange(750, 1250)
    # print("a", x, "=", a[x])
    # print("b", x, "=", b[x])
    # print("c", x, "=", c[x])
    # print("Pmin", x, "=", Pmin[x])
    # print("Pmax", x, "=", Pmax[x])

children = numpy.zeros(shape=(10, 10))  # 10 x 10 children matrix

crossoverPoint = [0] * 10  # crossover Points list

print("Power demand Pd= ", Pd)

# ----------------------- LOOP ----------------------

for i in range(200):

    print("--------- EPOCH ", i, "---------")
    HelperMatrix = numpy.zeros(shape=(0, 10))  # in the end used to copy connectedMatrices

    # ------------- Crossover -------------

    for C in range(5):
        crossoverPoint[C] = random.randint(1, 9)  # print("crossoverPoint for", i + 1, "th pare =", crossoverPoint[i])

    crossHelper = 0
    for g in range(0, 9, 2):
        children[g] = np.append(N[g][:crossoverPoint[crossHelper]], N[g + 1][crossoverPoint[crossHelper]:])
        # print("child = ", i, children[i], "crossoverPoint = ", crossoverPoint[crossHelper])
        children[g + 1] = np.append(N[g + 1][:crossoverPoint[crossHelper]], N[g][crossoverPoint[crossHelper]:])
        # print("child = ", i + 1, children[i + 1], "crossoverPoint = ", crossoverPoint[crossHelper])
        crossHelper = crossHelper + 1

    # ------------- Mutation -------------

    mutationGene1 = random.randint(0, 9)
    mutationGene2 = random.randint(0, 9)

    children[mutationGene1][mutationGene2] = random.randint(1, 32)  # 1 / 100
    children[mutationGene2][mutationGene1] = random.randint(1, 32)  # 2 / 100

    # ------------- Connected matrices  & PERMUTATION -------------
    ConnectedMatrices = np.concatenate((N, children))

    np.take(ConnectedMatrices, np.random.permutation(ConnectedMatrices.shape[0]), axis=0, out=ConnectedMatrices)

    # ------------- Calculations OLD -------------

    # for x in range(10):
    #     for y in range(10):
    #         # ------- for Parents ------
    #         P[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((N[x][y] - 1) / 31)
    #         K[x][y] = a[x] + b[x] * P[x][y] + c[x] * (P[x][y] ** 2)
    #
    #         # ------- for Children -------
    #         P_Ch[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((children[x][y] - 1) / 31)
    #         K_Ch[x][y] = a[x] + b[x] * P_Ch[x][y] + c[x] * (P_Ch[x][y] ** 2)

    # ------------- Calculations for Connected Matrices-------------

    for x in range(20):
        for y in range(10):
            # ------- for Parents ------
            Pconnected[x][y] = Pmin[y] + (Pmax[y] - Pmin[y]) * ((ConnectedMatrices[x][y] - 1) / 31)
            Kconnected[x][y] = a[y] + b[y] * Pconnected[x][y] + c[y] * (Pconnected[x][y] ** 2)

    for z in range(20):
        Econnected[z] = abs(sum(Pconnected[z]) - sum(ConnectedMatrices[z] * (Pconnected[z] ** 2)) - Pd)

        # ------- for Children -------
        # P_Ch[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((children[x][y] - 1) / 31)
        # K_Ch[x][y] = a[x] + b[x] * P_Ch[x][y] + c[x] * (P_Ch[x][y] ** 2)

    for z in range(20):
        KconnectedIndividual[z] = sum(Kconnected[z])
        # print("Individual Cost K[z]", z, KconnectedIndividual[z])

    # indexing [row][col]

    # First Epsilon 'E' then 'K' - Cost

    # ------------- Tournament Method-------------

    for z in range(0, 19, 2):
        # print("z =", z, "z + 1=", z + 1)
        if (Econnected[z] > Epsilon and Econnected[z + 1] > Epsilon):
            if (KconnectedIndividual[z] > KconnectedIndividual[z + 1]):
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])
            else:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
        elif Econnected[z] > Epsilon and Econnected[z + 1] < Epsilon:
            HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])
        elif Econnected[z] < Epsilon and Econnected[z + 1] > Epsilon:
            HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
        elif Econnected[z] < Epsilon and Econnected[z + 1] < Epsilon:
            if Econnected[z] < Econnected[z + 1]:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
            else:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])

    if i == 0:
        minFirstCost = min(KconnectedIndividual)
        print("\nFirst Epoch min configuration Cost:", minFirstCost)
        minFirstEpsilonSolution = min(Econnected)
        print("Minimal First Epoch Epsilon =", minFirstEpsilonSolution)

    N = HelperMatrix

print("\n\n\nThe final stage of the power plant operation")
print(N)

print("\nEpsilon list =", Econnected)
minEpsilonSolution = min(Econnected)
print("Minimal Epsilon =", minEpsilonSolution)
minCost = min(KconnectedIndividual)
print("Best Cost:", minCost)

print("\nEpsilon Difference=", minFirstEpsilonSolution - minEpsilonSolution)
print("Cost Difference= ", minFirstCost - minCost)
