import math

#Gradient有做padding因為題目的x, y coordinate of image的左上角是(1, 1)
Gradient = [
            [2, 3, 2, 1, 4, 2, 3, 5],
            [1, 0, 2, 1, 1, 7, 9, 1],
            [3, 3, 2, 1, 6, 9, 9, 2],
            [0, 1, 1, 12, 10, 15, 7, 0],
            [3, 4, 1, 9, 8, 6, 3, 1],
            [2, 3, 2, 12, 10, 10, 4, 2],
            [2, 1, 1, 0, 8, 12, 1, 1],
            [4, 3, 2, 2, 1, 3, 2, 2]
]
vertex = [
        [3, 1],
        [1, 6],
        [5, 7],
        [7, 4],
        [7, 1]
]
#上下左右方向
direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

#前一點往上下左右走最小的 energy
#先初始為0
Previous_Energy = [0, 0, 0, 0]

def calculate_E(i, j):
    Energy = [0, 0, 0, 0]
    #前一個點跑上下左右的energy
    for k in range(len(direction)):
        if Previous_Energy[k] != float('inf'):
            e = -pow((Gradient[vertex[i-1][0]+direction[k][0]][vertex[i-1][1]+direction[k][1]]), 2)
            e += pow(((vertex[i-1][0]+direction[k][0]) - (vertex[i][0]+direction[j][0])), 2) 
            e += pow(((vertex[i-1][1]+direction[k][1]) - (vertex[i][1]+direction[j][1])), 2)
            e += Previous_Energy[k]
            Energy[k] = e
        else:
            Energy[k] = float('inf')
    return Energy

temp_less_e = [0, 0, 0, 0]
def less_Energy(Energy, j):
    temp_less_e[j] = min(Energy)

def main():
    #從第一個點開始跑迴圈
    for i in range(len(vertex)):
        #上下左右四個方向
        for j in range(len(direction)):
            if i == 0:
                Previous_Energy[j] = -pow(Gradient[vertex[i][0]+direction[j][0]][vertex[i][1]+direction[j][1]], 2)
            else:
                if j == 0:
                    print("V" + str(i) + " to V" + str(i+1) + " :")
                if ((vertex[i][0]+direction[j][0]) <= 7) and ((vertex[i][0]+direction[j][0]) >= 0) and ((vertex[i][1]+direction[j][1]) >= 0) and ((vertex[i][1]+direction[j][1]) <= 7):
                    print("step" + str(j+1) + " :")
                    Energy = calculate_E(i, j)
                    less_Energy(Energy, j)
                    print(Energy)
                else:
                    temp_less_e[j] = float('inf')
        if i!= 0 :
            Previous_Energy[:4] = temp_less_e[:4]
            #print("Previous_Energy:", Previous_Energy)
            print("-------------------------------")

if __name__ == '__main__':
    main()