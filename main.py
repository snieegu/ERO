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
Epsilon = 100  # max strata

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

childs = numpy.zeros(shape=(10, 10))
# print("child before crossover")
# print(childs)
# print("child 1 before crossover")
# childs[0][2]= 7
# print(childs[0])
crossoverPoint = [0] * 10

print("Power demand Pd= ", Pd)

# ----------------------- LOOP ----------------------

for i in range(200):

    print("--------- EPOCH ", i, "---------")
    HelperMatrix = numpy.zeros(shape=(0, 10))

    # print("N wejsciowe:\n", N)

    # ------------- Crossover -------------
    for C in range(5):
        crossoverPoint[C] = random.randint(1, 9)
        # print("crossoverPoint for", i + 1, "th pare =", crossoverPoint[i])

    # print("N srodek 1:\n", N)

    crossHelper = 0
    for g in range(0, 9, 2):
        # childs[i] = [(N[i][0:crossoverPoint[i + 1]] + N[i+1][crossoverPoint[i + 1]:10])]
        # print(childs[i])
        # print("i =", i)
        childs[g] = np.append(N[g][:crossoverPoint[crossHelper]], N[g + 1][crossoverPoint[crossHelper]:])
        # print("child = ", i, childs[i], "crossoverPoint = ", crossoverPoint[crossHelper])
        childs[g + 1] = np.append(N[g + 1][:crossoverPoint[crossHelper]], N[g][crossoverPoint[crossHelper]:])
        # print("child = ", i + 1, childs[i + 1], "crossoverPoint = ", crossoverPoint[crossHelper])
        # parents = parents + tuple([(parents[0][0:cross_point + 1] + parents[1][cross_point + 1:6])])
        crossHelper = crossHelper + 1

    # print("powstale dzieci 2:\n", childs)

    # print("Post CrossOver Childs")
    # print(childs)

    # ------------- Mutation -------------

    mutationGene1 = random.randint(0, 9)
    # print("mutationGene1= ", mutationGene1)
    mutationGene2 = random.randint(0, 9)
    # print("mutationGene2= ", mutationGene2)
    childs[mutationGene1][mutationGene2] = random.randint(1, 32)
    # print(childs)

    # ------------- Connected matrices  & PERMUTATION -------------
    ConnectedMatrices = np.concatenate((N, childs))

    # print("connected matrices\n", ConnectedMatrices)

    np.take(ConnectedMatrices, np.random.permutation(ConnectedMatrices.shape[0]), axis=0, out=ConnectedMatrices)

    # print("connected shuffeled matrices\n", ConnectedMatrices)

    # ------------- Calculations -------------

    # for x in range(10):
    #     for y in range(10):
    #         # ------- for Parents ------
    #         P[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((N[x][y] - 1) / 31)
    #         K[x][y] = a[x] + b[x] * P[x][y] + c[x] * (P[x][y] ** 2)
    #
    #         # ------- for Children -------
    #         P_Ch[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((childs[x][y] - 1) / 31)
    #         K_Ch[x][y] = a[x] + b[x] * P_Ch[x][y] + c[x] * (P_Ch[x][y] ** 2)

    # ------------- Calculations for Connected metrices-------------
    # print("Połączone N i dzieci srodek 3:\n", ConnectedMatrices)

    for x in range(20):
        for y in range(10):
            # ------- for Parents ------
            Pconnected[x][y] = Pmin[y] + (Pmax[y] - Pmin[y]) * ((ConnectedMatrices[x][y] - 1) / 31)
            Kconnected[x][y] = a[y] + b[y] * Pconnected[x][y] + c[y] * (Pconnected[x][y] ** 2)

    # print('Connected matrix P')
    # print(Pconnected)

    # print('Connected matrix K')
    # print(Kconnected)

    # print("Epsilon osobników - Connected")
    for z in range(20):
        Econnected[z] = abs(sum(Pconnected[z]) - sum(ConnectedMatrices[z] * (Pconnected[z] ** 2)) - Pd)
        # print("Epsilon osobnika | E[z] |", z, Econnected[z])

        # ------- for Children -------
        # P_Ch[x][y] = Pmin[x] + (Pmax[x] - Pmin[x]) * ((childs[x][y] - 1) / 31)
        # K_Ch[x][y] = a[x] + b[x] * P_Ch[x][y] + c[x] * (P_Ch[x][y] ** 2)

    for z in range(20):
        KconnectedIndividual[z] = sum(Kconnected[z])
        # print("Koszt osobnika K[z]", z, KconnectedIndividual[z])

    # ------------- Printing -------------
    #
    # # indexowanie [wiersz][kolumna]
    #
    # print('Power (P) - matrix')
    # print(P)
    #
    # print('Cost (K) - matrix')
    # print(K)
    #
    # for g in range(10):
    #     print("suma mocy P[z]", sum(P[g]))
    #
    # for g in range(10):
    #     print("suma stopni N[z]", sum(N[g]))
    #
    # print("Epsilon osobników")
    # for z in range(10):
    #     E[z] = abs(sum(P[z]) - sum(N[z] * (P[z] ** 2)) - Pd)
    #     print("Epsilon osobnika | E[z] |", z, E[z])
    #
    # for g in range(10):
    #     print("Koszta osobikow K[z]", g, sum(K[g]))

    # porówbujemy najpierw Epsilon 'E' a później 'K' - Koszt

    # ------------- Matching by tournament method -------------

    # ------------- Tournament Method-------------
    # print("Połączone N i dzieci srodek 4:\n", ConnectedMatrices)

    for z in range(0, 19, 2):
        # print("z =", z, "z + 1=", z + 1)
        if (Econnected[z] > Epsilon and Econnected[z + 1] > Epsilon):
            if (KconnectedIndividual[z] > KconnectedIndividual[z + 1]):
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])
                # ConnectedMatrices = np.delete(ConnectedMatrices, z, 0)
            else:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
                # ConnectedMatrices = np.delete(ConnectedMatrices, z + 1, 0)
        elif Econnected[z] > Epsilon and Econnected[z + 1] < Epsilon:
            HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])
            # ConnectedMatrices = np.delete(ConnectedMatrices, z, 0)
        elif Econnected[z] < Epsilon and Econnected[z + 1] > Epsilon:
            HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
            # ConnectedMatrices = np.delete(ConnectedMatrices, z + 1, 0)
        elif Econnected[z] < Epsilon and Econnected[z + 1] < Epsilon:
            if Econnected[z] < Econnected[z + 1]:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z]])
                # ConnectedMatrices = np.delete(ConnectedMatrices, z + 1, 0)
            else:
                HelperMatrix = np.vstack([HelperMatrix, ConnectedMatrices[z + 1]])
                # ConnectedMatrices = np.delete(ConnectedMatrices, z, 0)

    # print("wyjsciowe helperMatrix: \n", HelperMatrix)

    N = HelperMatrix
    # sleep(3)
    print("Epsilon osobnikow wyjsiowe", Econnected)
    print("N wyjsciowe:\n", N)

print("\n\n\nThe final stage of the power plant operation")
print(N)

print("Epsioln list =", Econnected)
minEpsilonSolution = min(Econnected)
print("Minimal Epsiol =", minEpsilonSolution)
