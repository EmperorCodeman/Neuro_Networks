import numpy as np
from PIL import Image

"""
    Realizations:
        If the input does not have geometric meaning. Ie if the positions of cells has no meaning then input should be 1D. 
            2D is a batch operation of 1D matricies where each column of parshed input has no interaction with other columns no matter the depth. Unless you transpose after each layer. which is bad
        For geometric data, we need the relationship between regions. Thus the filters. Filters are nXmX3 for images. Use padding or center of image has more weight. Use n and m as odd so filter centers on pixel
            Then output forms a tensor which a new tensor filter goes over post max pool
        Dnn is for logic
            1D input is independent vars, 1d output from layer 1 is dependent vars. Where each var factors all inpdependent vars. subsiquent layers abstract dependent vars further into higher and higher level inference
                example, grades on everything for student. Layer 1 output is hw average, quiz average, test average, subject competincies. layer 2 ouput is course grade, carrier path readiness... etc. Layer 3 output is recommeded action for student  
            Thus a Dnn in classical function theory is called a Composition of functions. 
                where each neuron is a function and each next layer is a compositon of those functions
        A Dnn layer is a transformation from one feature space to another. 
            thus after the network is trained. Simply its architicture by doing principal component analysis maybe
            or as you understand. Use linear algebra to remove duplicate rows of layer(perceptrons). ie two rows that are not independent. 
            Remove any row that can be composed from other rows.  
        
    We are seeing various ways of optimizing the weights 
    Systems of non linear equations. Set gradient of loss function to zero and solve. 
    Interpolate the loss function with cubic spline interpolation then solve for interpolated roots 
    gradient descent 

    maybe I am seeing decreased disparity between training and testing accuracy as dataset sizes increase

    I audited this. Average loss does indeed decrease with smaller batch size. However this has no value in terms of testing accuracy. 
        The residual gets way smaller with smaller batch size. I think this is because the gradient is more focused on less columns. 
            Printing loss before step doesnt explain this. I dont know why the loss gets smaller with batch size.  
    
    Bias theory:
        Bias is a translation of the origin in respective feature space
        Bias increases a networks tolerance of mislabeled data in training
            I think this is because the nested origin is resolvable due to large sample size. 
            By having a nested origin we are deviating from the correct origin. Making default error less
        Adding bias without any bias gradient had no effect 

    Neuro nets profit from outliers, unlike stats. We the brain learn better from outliers and so do they  

    TODO Use induction to differentiate any dnn
    Include a classification neuron for unclassified so that the net is not forced to classify stuff it cannot recognize 
"""

class ACTIVATIONS:

    @staticmethod
    def reLU(dendritic_input):
        #   Build the activation Function
        #   Model the Neuro Synapse with reLU function. If dendritic input is greater than 0 then fire neuron
        return dendritic_input * (dendritic_input > 0) 

    @staticmethod
    def reLU_primed(dendritic_input):
        #   The derivitave of pairwise function is a pairwise derivative. Simply its the derivitive of each of the functions it directs to 
        #   In this case. x or 0. Thus its derivitive is 1 or 0 
        return np.ones_like(dendritic_input) * (dendritic_input > 0)

    @staticmethod
    def softmax(dendritic_input):
        e_to_x = np.exp(dendritic_input)
        return e_to_x / np.sum(e_to_x, axis=1)
        
    @staticmethod
    def softmax_primed(dendritic_input):
        #   l-1 prime used inductively. thus last layers activation primed unneeded
        raise Exception("Back propagation induction allows us to skip the first activations prime")
        #   Using quotient rule we arive at this. Neglecting jacobian concept
        e_to_x = np.exp(dendritic_input)
        column_sum = np.sum(e_to_x, axis=1)
        coeff = e_to_x / column_sum**2
        return coeff*(column_sum - e_to_x)
        
    @staticmethod
    def none(dendritic_input):
        return dendritic_input

class LOSS_FUNCTIONS:
    
    @staticmethod
    def mean_squared_error(dnn, batch, supervision):
        #   The expectation is the oupute of the net
        residual = dnn.feed_forward(batch) - supervision
        return np.sum( residual**2 ) / residual.shape[1] 

    @staticmethod
    def mean_squared_error_primed(dnn, batch, supervision):
        return (2/batch.shape[1]) * dnn.feed_forward(batch) - supervision

    @staticmethod
    def cross_entropy(dnn, batch, supervision):
        #   Cross entropy requires inputs as probabilities. No negatives allowed
        #   Google cross entropy for its theory
        supervision = supervision == 1
        return np.sum(-np.log(dnn.feed_forward(batch)[supervision])) / supervision.shape[1]

    @staticmethod
    def cross_entropy_primed(dnn, batch, supervision):
        #   When I did the gradient I got the below. 
        #coeff = -supervision * supervision.shape[1]
        #return coeff / dnn.feed_forward(batch) 
        #   Universal Proved gradient
        pass

class DNN:
    def __init__(self, neurons_per_layer, data,\
            loss_function_and_prime=[LOSS_FUNCTIONS.mean_squared_error, LOSS_FUNCTIONS.mean_squared_error_primed],\
            hidden_layers_activation_and_prime=[ACTIVATIONS.reLU, ACTIVATIONS.reLU_primed],\
            final_activation_and_prime=[ACTIVATIONS.softmax, ACTIVATIONS.softmax_primed]):
        """
            We initiate our DNN with supervised learning. Passing it our data class. 
            We also pass it a architecture shape. Neurons per layer infers the architecture by liner algebra syntax
            We express layers as a single variable.
                Using this substituition we are able to perform abstract calculus that is independent of layers shape
                Thus chose any shape architecture and we can infer the gradients
            Neurons/Layer: Parse first layer first, then all hidden layers. 
                Last layer is defined by data set. so do not parse it in Neurons per layer arg
            Note the first column is the bias. Which is translation of the origin. first row of input is always 1
            We use reLU as the only activation function, always activating between hidden layers 
        """

        def initialize_layer(neurons_count, last_layers_rows):
            #   Initialize weights as from uniform distribution between -1, and 1
            return np.random.uniform(-1, 1, last_layers_rows*neurons_count).reshape(neurons_count, last_layers_rows)

        def initialize_biases(layers):
            biases = []
            for layer in layers:
                biases.append( np.random.uniform(-1, 1, layer.shape[0]).reshape(layer.shape[0], 1) )
            biases = np.zeros_like(biases) #    Remove biases
            return biases

         #  Attach data set to DNN. You can switch data set at any time for transfer learning
        self.data = data

        #   Initialize all layers weights as np objects with the shapes as given from neurons per layer
        input_layer = initialize_layer(neurons_per_layer[0], data.train_data.shape[0]) #  Plus one is for bias
        layers = [input_layer]
        for i, neural_count in enumerate(neurons_per_layer[1:-1]): #  Hidden layers 
            layers.append(initialize_layer(neural_count, neurons_per_layer[i]))
        #   Init the final layer. It conforms its shape entirely and is not programable
        layers.append(initialize_layer(data.train_supervision.shape[0], neurons_per_layer[-1]))
        self.layers = layers

        #   Data storage for layers outputs
        self.flows = [None] * len(layers)

        #   create biases
        self.biases = initialize_biases(layers)

        #   Tie activation functions and their derivities to dnn
        self.hidden_activation, self.hidden_activation_primed = hidden_layers_activation_and_prime
        self.final_activation, self.final_activation_primed = final_activation_and_prime

        #   Tie Loss 
        self.get_loss, self.loss_primed = loss_function_and_prime 

    def feed_forward(self, input, forward_propagating=False):
        #   Parse independent vars, ie observation into the networks first layer and process activation func if network has hidden layers 
        if forward_propagating:
            #   Save each output per layer for backpropagating. optimization 
            if len(self.layers) > 1: #  if only 1 layer then the final layer is the only layer.  
                self.flows[0] = self.layers[0] @ input + self.biases[0] #    Backpropagation uses the weighted input unactivated
                flow = self.hidden_activation( self.flows[0] )
            else:
                self.flows[0] = self.layers[0] @ input + self.biases[0]
                flow = self.final_activation(  self.flows[0] ) 
                return flow    

            #   Flow and use activation function for all hiden layers
            for i, layer in enumerate(self.layers[1:-1]):
                self.flows[i+1] = layer @ flow + self.biases[i] 
                flow = self.hidden_activation( self.flows[i+1] )
                
            #   All activations can be different, we only change the final activation for simplicity 
            self.flows[-1] = self.layers[-1] @ flow  + self.biases[-1] 
            flow = self.final_activation( self.flows[-1] )
 
        else: # I split this for optimization. Only check the condition once. Dont rewrite flows uneeded  

            if len(self.layers) > 1: #  if only 1 layer then the final layer is the only layer.
                flow = self.hidden_activation( self.layers[0] @ input + self.biases[0] )
            else:
                flow = self.final_activation(self.layers[0] @ input + self.biases[0] )
                return flow 

            #   Flow and use activation function for all hiden layers
            for i, layer in enumerate(self.layers[1:-1]):
                flow = self.hidden_activation( layer @ flow  + self.biases[i] )
                
            #   Forgo activation function on the last function 
            flow = self.final_activation(self.layers[-1] @ flow + self.biases[-1] )
    
        return flow # flow.reshape(len(flow),1) if batch size 1 reshape needed
        
    def get_accuracy(self, batch, batch_supervision):
        #   Only for classification. Take Pass batches from the testing data partition
        batch_size = batch.shape[1]
        return sum(np.argmax(self.feed_forward(batch), axis=0) == np.argmax(batch_supervision, axis=0)) / batch_size

    def get_gradient(self, batch, supervision):
        #   Old method
        #gradient_layer_1 = self.layers[1].transpose() @ loss_primed @ self.hidden_activation_primed( batch.transpose() ) 
        #gradient_layer_2 = loss_primed @ self.hidden_activation(layer_1_output.transpose())

        #   2 layers only supported currently 
        #   TODO NOTE each layer can be optimized on its own thread. no lock needed. the other layers will be updated as they do. residual only updated once per master loop 
        loss_primed = self.loss_primed(self, batch, supervision) 
        
        self.feed_forward(batch, forward_propagating=True) #    Store the layers outputs to class
        layer_1_output = self.flows[0]   #self.layers[0] @ batch

        gradient_layer_1 = self.layers[1].transpose() @ loss_primed @ self.hidden_activation_primed( batch.transpose() ) 
        gradient_layer_2 = loss_primed @ self.hidden_activation(layer_1_output.transpose())


        #gradient_layer_1 = 4
        #gradient_layer_2 = 0
        return [gradient_layer_1, gradient_layer_2]

    def fit(self, batch_size=100, epochs_limit=100):
        #   Doubling batch size doubles the speed of a epoch. Smaller batch size, slower epoch but less general

        def delta_loss(step_magnitude, last_loss, buffered_network):
            #   Only a temp update of the layer
            self.layers = [buffered_network[i] - step_magnitude*layers_gradient for i, layers_gradient in enumerate(gradient)]
            perspective_loss = self.get_loss(self, batch, batch_supervision)
            delta_loss = perspective_loss - last_loss #  Note: The loss is always positive. Thus delta loss is [0,original_loss]
            return delta_loss, perspective_loss

        def get_parabola_min(three_input, three_outpute):
            """
                A parabala is: Y = ax^2 + bx + c
                Algebra to vertex form is: Y = a(x - h)^2 + k   Where h is the x of the vertex and k is its y
                To solve we have 2 equations with two unknowns for polynomial form, because C is always 0 due to 0 step causing 0 delta loss ie 0 y intercept  
                    The first point is always (0,0) thus C the y intercept is always 0 in this case
                We set the system, invert it then matrix multiply to get a and b. Then use algebra to solve for the vertex
            """
            """
                #   This is the linear algebra process to obtain the vertex
                system_of_equations = np.array([\
                    [three_input[1]**2, three_input[1]],\
                    [three_input[2]**2, three_input[2]]
                    ]) 
                a_and_b = np.linalg.inv(system_of_equations) @ three_outpute[1:]
                vertex_x = -a_and_b[1] / (2* a_and_b[0]) 
            """

            #   Algebracially we can solve for the vertex using variables. So no need to process inverse matrix ect. 
            x2, x3 = three_input[1:]
            y2, y3 = three_outpute[1:]
            denom = x2*x3*(x2 - x3)
            A = (x3 * y2 + x2 * (- y3)) / denom
            B = (x3**2 * (- y2) + x2**2 * (y3)) / denom
            if denom == 0 or A == 0: return .0001  #   Avoid division by zero. Should not be needed 
            vertex_x = -B / (2*A)
            return vertex_x

        last_step = 1
        test_sample_size = 300
        test_batch = self.data.test_data[:, 0:test_sample_size]
        test_batch_supervision = self.data.test_supervision[:, 0:test_sample_size]
        probability_of_printing_readout_per_iter = 100

        print("\n\n\n\t\t\t\t\t\tNeurons: " + str(self.layers[0].shape[0]) + "\t\t Batch Size: " + str(batch_size))
        for epoch in range(epochs_limit): 
            inter_epoch_iteration = 1
            print("\n\n\t\t\t EPOCH: " + str(epoch+1) + "\n------------------------------------------------------------------------\n")
            for i in np.random.permutation(np.arange(self.data.train_data.shape[1]-batch_size)): 
                """
                    i = batch iteration
                    Iterate over the entire training data_set every epoch. 
                    The batch size creates the size of the window. From there we increment each the window till the end of the set
                    We optimize the weights every iteration, and we optimize the step size every iteration
                    Use of np.random is explained below
                    Lastly we randomize the sequence of iteration. I thought of this personally by this thinking:
                        If the windows are sequential then overfitting will result because all but one of the elements of the last iteration will be the same. 
                        This will cause the net to over fit to that area of the set rather then have a ballanced decent from random windows 
                        Thus we randomize the sequence of windows 
                """
                batch = self.data.train_data[:, i:i+batch_size]
                batch_supervision = self.data.train_supervision[:, i:i+batch_size]
                gradient = self.get_gradient(batch, batch_supervision) #    Gradiant of all layers
                last_loss = self.get_loss(self, batch, batch_supervision) #   Compaire each potential step size against loss of no step size in delta loss  
                """   
                    Now we do a line search inorder to find a productive step size for the gradient descent. 
                    F(step size) = change in loss from last epoch due to step size 
                        We want to find the minumum. Where negative change is good. B - A                        
                        We know that the change in loss for 0 step size is 0.
                            If we find 2 more points we can create a parabala 
                            Note: We know that the parabala starts with a negative slope by gradient theory. Tangent line at point of tangential
                            Note: If we have more than three points we discard the others. Larger sample of points causes overfitting to high degree polynomial which will create problamatic extremas  
                            Thus we chose the three points closest to zero
                            We loop with our exit condition being finding a negative value of F(step size). 
                        Lastly we step 1.5 times the magnitude of known productive step. 
                        This gives [0, known_neg_step, 1.5*known_neg_step] as our parabola points

                """
                
                buffered_network = self.layers 
                d_loss, perspective_loss = delta_loss(last_step, last_loss, buffered_network)
                loop_count = 0
                while (d_loss >= 0): #  While there is no improvement with step length, half the step and check again
                    last_step *= .5 #   Half step. Remember, tangent to 0 step is always negative delta loss. So there is always a solution
                    #self.layers[i] = buffered_layer
                    d_loss, perspective_loss = delta_loss(last_step, last_loss, buffered_network)
                    loop_count += 1 #   In the event that the network is perfectly fit to the data there will be no improvment possible. Break
                    if loop_count == 30:
                        last_step = 0
                        break
                    if d_loss == 0:
                        if np.sum(self.feed_forward(batch)) == 0: 
                            raise Exception("Network is Zeroed out from reLU")
                            #   The reLU function is causing the losses to equal because the output equals zero pre and post descent
                            #   This is appears to be a fatal error. Solution may be to use leaky reLU. if < 0 then x *= -.001  
                        #   Unknown logic: for some reason the small gradient is producing a the same loss as the last epoch
                        last_step = .0001 # Reset the step to small value
                        d_loss, perspective_loss = delta_loss(last_step, last_loss, buffered_network)
                        
                if last_step == 0: 
                    #probability_of_printing_readout_per_iter = int(probability_of_printing_readout_per_iter/2)
                    #   The network is now fit to the training data pretty well. So print when fitting occures more often
                    #   Reason is that new batches can still provide fitting oppurtunities
                    #   Insure that steps always lower loss for that batch
                    continue

                #   We now know that there delta loss is negative. Lets optimize the step size further with parabula then finalize step
                parabala_points_x = [0, last_step,   last_step*1.5] 
                upstep_delta_loss, upstep_loss = delta_loss(parabala_points_x[2], last_loss ,buffered_network)
                parabala_points_y = [0, d_loss, upstep_delta_loss]
                parabala_points_loss = [last_loss, perspective_loss, upstep_loss]

                #   Parabalas need three non linear points. X points will always be different
                if parabala_points_y[1] == parabala_points_y[2]: 
                    best_step_index = 1 #   This step is known to be good. Parabula didnt help
                else:
                    #   Now using the the three points find the min of a parabala. delta_loss(step_size) only min
                    parabala_step = get_parabola_min(parabala_points_x, parabala_points_y)
                    delta_loss_parabala, parabala_loss = delta_loss(parabala_step, last_loss, buffered_network)
                    
                    #   Find the min of all steps, then use it as final step. This to insure that steps can only lower loss
                    parabala_points_x.append(parabala_step)
                    parabala_points_y.append(delta_loss_parabala)
                    parabala_points_loss.append(parabala_loss)
                    best_step_index = np.argmin(parabala_points_y)

                if parabala_points_x[best_step_index] == 0: 
                    raise Exception("Only negative delta loss should be at this point. Logic bad rewrite")

                #   Finalize Step and prepair for next iteration
                best_known_step = parabala_points_x[best_step_index] 
                last_step = best_known_step
                last_loss = parabala_points_loss[best_step_index]
                self.layers = [buffered_network[i] - best_known_step*layers_gradient for i, layers_gradient in enumerate(gradient)]
                if (i % probability_of_printing_readout_per_iter) == 0:                  
                    test_accuracy = np.round(self.get_accuracy(test_batch, test_batch_supervision), 2)
                    train_accuracy = np.round(self.get_accuracy(batch, batch_supervision), 2)
                    print("\tIteration: " + str(inter_epoch_iteration) + "\t Step Size: " + f'{last_step:.2E}' + "\t Training Loss: " + str(np.round(last_loss, 2))\
                        + "\t Training Accuracy: " + str(train_accuracy) + "\t Testing Accuracy: " + str(test_accuracy))
                        
                    if test_accuracy > 0.95: return # trained
                inter_epoch_iteration += 1

    def temp_fit(self, epochs, data, supervision):
        step_size = 1
        for epoch in range(epochs):
            print("\n\n\t\t\t EPOCH: " + str(epoch+1) + "\n------------------------------------------------------------------------")
            gradients = self.get_gradient(data, supervision)
            for i, layers_gradient in enumerate(gradients):
                gradients_magnitude = np.sum(layers_gradient**2)**.5
                if gradients_magnitude == 0: continue
                layers_gradient = layers_gradient / gradients_magnitude #   Normalize
                self.layers[i] -= step_size * layers_gradient
                loss = self.get_loss(data, supervision)
                print("Layer: " + str(i+1) + "\n\tLoss: " + str(loss))
                if loss < 0.01: return 
                
class MNIST:
    def __init__(self) -> None:

        def normalize_tensor(tensor):
            #   Normalize the inpute to keep it close to activation value 0. We change data structure to float
            active_pixels = tensor != 0
            #active_pixels[0,:] = False #    Bias 1 left out
            z_scores_of_pixels = (tensor[active_pixels] - np.average(tensor[active_pixels])) / np.std(tensor[active_pixels])
            tensor = np.zeros_like(tensor, dtype=float) #   This changes the data structure from uint8 to float
            tensor[active_pixels] = z_scores_of_pixels
            # tensor[0,:] = 1 # Convert Label to 1. This will always multiply times the bias in the respective nueron
            return tensor

        def load_data_from_csv(file_location):
            """
                CSV Format: row at image. row 1 as header. Column one as solution lable. 
            """
            #   
            data = np.genfromtxt(file_location, delimiter=',', dtype=np.uint8)[1:] #    First Row is Gumbo     
            #   Pause here and call images if you want. Before normalization
            #MNIST.show_image_from_row(data.test_data, 0)

            #   Solution as a one hot vector. In MNIST we have 0-9 as labels. We put the position of the output neurons as the value of the label
            supervision = np.zeros(shape=(10, len(data))) # Ten Labels X Data set rows 
            supervision[data[:,0], range(len(data))] = 1

            #   This is the last point you can draw img without reversing z score to pixels

            #   A instance is a column as input in matrix multiplication thus the transpose. Remove the label column
            data = data[:,1:].transpose() 
            data = normalize_tensor(data) # This will convert to z scores from data 
            
            return data, supervision

        #self.train_data, self.train_supervision = load_data_from_csv('data_sets/mnist_train.csv')
        #self.test_data, self.test_supervision =   load_data_from_csv('data_sets/mnist_test.csv')
        self.train_data, self.train_supervision = load_data_from_csv('data_sets/mnist_test.csv')
        self.test_data, self.test_supervision = self.train_data[:,9000:], self.train_supervision[:,9000:] 
        self.train_data, self.train_supervision = self.train_data[:,:9000], self.train_supervision[:,:9000]
        
    @staticmethod
    def show_image_from_row(data, row):
        image = data[row,1:].reshape(28,28) #   Skip label
        #   Image will not work unless dtype is uint8
        image = np.repeat(image[:,:,np.newaxis], 3, axis=2).astype(np.uint8) #   RGB repeat for black and white                
        Image.fromarray(image, 'RGB').show()
        #Image.fromarray(image, 'RGB').save("temp/random.jpg")
        print("\nLable for image: " + str(data[row,0]))

data = MNIST()

neurons_per_layer = [1000] # First layers neuron count. Second layer defined implicitly
dnn = DNN(neurons_per_layer, data=data, final_activation_and_prime=[ACTIVATIONS.none, ACTIVATIONS.none])

dnn.fit(batch_size=10, epochs_limit=5)
dnn.fit(batch_size=100, epochs_limit=3)
dnn.fit(batch_size=1000, epochs_limit=5)

 
#   bias's were added wrong. remove the old way and add them correctly. 

#   Optimaization. Save feed forward to memory in dnn and access it so no need to recall it in loss 

i = 2


"""
    Mini lecture:
        Explain bias and how his solution created no need to change anything with bias
            bias as a translation of the origin in feature space of layer
        Feed forward as generalization, Feed backward as extrapolation, or dreaming 
            Bi directional learning 
        Goal: 
            Jan starting school
            Full Dnn
            Hypothesis testing and thoery. Lets optimize batch size 
            Data as a assest 
            thought vectors and translation. One hot vector classification to thought vector classification
            Memory short and long. brief modeling 
                rnn review. 
            Black Jack ok. Job interview had me do it. 
                Dnn as ai
                synthetic data set
                include var for personality of player. risk aversion
                    personality of user unknown to net. 
                    memory needed inorder to case player and mesure player personality 

"""