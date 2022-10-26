import numpy as np

"""
    Realizations:
        If the input does not have geometric meaning. Ie if the positions of cells has no meaning then input should be 1D. 
            2D is a batch operation of 1D matricies where each column of parshed input has no interaction with other columns no matter the depth. Unless you transpose after each layer. which is bad
        For geometric data, we need the relationship between regions. Thus the filters. Filters are nXmX3 for images. Use padding or center of image has more weight. Use n and m as odd so filter centers on pixel
            Then output forms a tensor which a new tensor filter goes over post max pool
        Dnn is for logic
            1D input is independent vars, 1d output from layer 1 is dependent vars. Where each var factors all inpdependent vars. subsiquent layers abstract dependent vars further into higher and higher level inference
                example, grades on everything for student. Layer 1 output is hw average, quiz average, test average, subject competincies. layer 2 ouput is course grade, carrier path readiness... etc. Layer 3 output is recommeded action for student  
             

    We are seeing various ways of optimizing the weights 
    Systems of non linear equations. Set gradiant of loss function to zero and solve. 
    Interpolate the loss function with cubic spline interpolation then solve for interpolated roots 
    Gradiant descent 
        while f - alpha*grad > f - .5*alpha*grad 
            alpha /= 2
        quadratic interpolation or cubic interpolation 

    Include a classification neuron for unclassified so that the net is not forced to classify stuff it cannot recognize 
"""

input = np.arange(300,303).reshape(3,1)
layer_1 = np.reshape(np.arange(9, dtype=float), (3,3))
goal = np.arange(6,9).reshape(3,1)
step_magnitude = .000001

def get_gradiant(layer, input, goal):
    input = np.repeat(input,3, axis=1) 
    residual = layer @ input - goal
    return -input.transpose() * residual

epochs = 100
for i in range(epochs):
    layer_1 += step_magnitude * get_gradiant(layer_1, input, goal)
    print( np.round(layer_1 @ input, 2), "\n")
 
# add another 3,3 layer. 1,3 = 1,3 @ 3,3 @ 3,3 
# next automate step magnitude







# loss = get_loss
# parameters = 0

i = 1

# def get_gradiant(loss, input, data_in, data_goal):
#     loss
#     pass

# get_gradiant(loss, parameters)
#original_loss = get_loss(layer_1, input, goal)
#print( np.round(layer_1 @ input, 2), "\n")
def get_loss(layer, input, goal):
    return np.sum(((layer @ input - goal)**2)/2)
    #return np.array([(((row@data_in) - data_goal)**2)/2 for i, row in enumerate(input)])    

#grad = -input.transpose() * residual
    #normalized_grad = grad / np.sum(grad**2)**.5
    #step_magnitude = 1 / np.average(residual)**2#np.repeat(residual, 3, axis=1)
    #scaled_grad = step_magnitude * normalized_grad
    #step_magnitude = 10000000  
    #scaled_grad = grad * step_magnitude
    #return scaled_grad