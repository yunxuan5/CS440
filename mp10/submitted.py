'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    # raise RuntimeError("You need to write this part!")
    # M, N = model.M, model.N
    # P = np.zeros((M, N, 4, M, N))

    # for r in range(M):
    #     for c in range(N):
    #         if model.T[r, c]:
    #             continue

    #         for a in range(4):
    #             for r_next, c_next, prob in model.D[r, c, a]:
    #                 P[r, c, a, r_next, c_next] = prob

    # return P

    rows, cols = model.M, model.N
    P = np.zeros((rows, cols, 4, rows, cols))

    for r in range(rows):
        for c in range(cols):
            if model.T[r, c]:
                P[r, c, :, :, :] = 0
            else:
                for a in range(4):
                    # Calculate the next cell based on action a
                    if a == 0:  #left case
                        if c == 0 or model.W[r,c - 1]:            #stay indended, move left
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r,c - 1] += model.D[r,c,0]
                        
                        if r == rows - 1 or model.W[r + 1,c]:     #counter-clockwise, move down
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r + 1,c] += model.D[r,c,1]
                        
                        if r == 0 or model.W[r - 1,c]:            #clockwise, move up
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r - 1,c] += model.D[r,c,2]

                    elif a == 1:  #up case
                        if r == 0 or model.W[r - 1,c]:            #stay indended, move up
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r - 1,c] += model.D[r,c,0]

                        if c == 0 or model.W[r,c-1]:              #counter-clockwise, move left
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r,c - 1] += model.D[r,c,1]

                        if c == cols - 1 or model.W[r,c + 1]:     #clockwise, move right
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r,c + 1] += model.D[r,c,2]

                    elif a == 2:    #right case
                        if c == cols - 1 or model.W[r,c + 1]:     #stay indended, move right
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r,c + 1] += model.D[r,c,0]
                            
                        if r == 0 or model.W[r - 1,c]:            #counter-clockwise, move up
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r - 1,c] += model.D[r,c,1]
                            
                        if r == rows - 1 or model.W[r + 1,c]:     #clockwise, move down
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r + 1,c] += model.D[r,c,2]

                    elif a == 3:    #down case
                        if r == rows - 1 or model.W[r + 1,c]:     #stay indended, move down
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r + 1,c] += model.D[r,c,0]
                            
                        if c == cols - 1 or model.W[r,c+1]:       #counter-clockwise, move right
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r,c + 1] += model.D[r,c,1]
                            
                        if c == 0 or model.W[r,c-1]:              #clockwise, move left
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r,c-1] += model.D[r,c,2]

    return P

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")
    rows, cols = model.M, model.N
    U_next = np.copy(U_current)
    for r in range(rows):
        for c in range(cols):
            utility = np.zeros(4)
            for a in range(4):
                utility[a] = np.sum(P[r,c,a,:,:] * U_current)
            max_utility = np.max(utility)
            U_next[r, c] = model.R[r,c] + model.gamma * max_utility
    return U_next


def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    rows, cols = model.M, model.N
    U = np.zeros((rows, cols))
    while(True):
        U_next = update_utility(model, compute_transition_matrix(model), U)
        if np.max(np.abs(U_next - U)) < epsilon:
            break
        U = U_next
    return U

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
