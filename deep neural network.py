import numpy as np_ #   Numpy uses the CPU. Its slower, but the library appears more stable, and better exception handeling 
from PIL import Image, ImageOps, ImageFilter # Used to debug images. Optional 
import cupy as np # CuPy deffinitly speeds up operations. For example it does not explain when overflow occures making tracing harder then Numpy
import time #   Used to time program speed. Optional 
import shelve  
import os
import copy

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
            Note this is Average loss note total loss
    
    Bias theory:
        Bias is a translation of the origin in respective feature space
        Bias increases a networks tolerance of mislabeled data in training
            I think this is because the nested origin is resolvable due to large sample size. 
            By having a nested origin we are deviating from the correct origin. Making default error less
        Adding bias without any bias gradient had minor negative effect
        I think:
            Biases help because they move the y intercept to the average weight of without biases   
            From here the weights ie slope, can tune the output. y = mx + b.
                Meaning the variance of the weights decreases with a bias offset
                Thus the range of the weights decreases, and greater generalization is found
            Note any y can be found from any none zero input with a given weight. However, deviating from a y offset decreases the range of the slopes to achieve the desired y's
        Note also:
            Biases control the sensitiviy of the activation:
                Meaning with a large negative bias we are making a neuron need larger input before it activates
                    This means. Biases give neurons specialization of labor. Asymetric activation thresholds

    Neuro nets profit from outliers, unlike stats. We the brain learn better from outliers and so do they  

    If the training loss goes to zero and the testing loss is above zero then increase training sample size. As training sample size goes to infinity it converges with the testing loss
    You should always be able to fit to training perfectly no matter batch size or your nets wrong. 
    As training set size goes to infinity, if training loss is zero, then the descrepency between training loss and testing loss will go to zero

    Inverse of network:
        The feed forward direction of a network from large input to finish with small output or classification can be surmized as: Extrapolation => Generalization, if we invert the network we have: Generalization => Extrapolation
        This shows the importance of bi-directional learning. We learn from our extrapolation, inorder to generalize. ie a finit state machine maybe

    Optimization:   
        GPU with cupy speed up the program 19x fold
        TODO
            multi thread the find step function 
            launch x threads each iteration. Each thread finds its own gradient*step then returns to master thread. The average of these forms the next sequential gradient step. 
            Below seems unneeded because the fitting is not that hard. The problem is closing the difference between train and test. A difference that appears to converge to zero with sample size
                Step different lengths for each layers weights and biases. This would create a massively complex step. Notice the last layers have more magnitude in their gradients. thus they should have smaller steps
                    To find their lengths. do mesh search... hyperdimensional line search. Find values and store them in a array. Then decay the array over iterations and use a multinomial to select the weights from there. 

    
        About the Program: 

    We can hone saved nets or build new ones with any shape. 
        We can engage a lock so that transfer learning or any learning for that matter only improves performance on the esoteric desired data set. 
            Note we never access the live feed for any reason other than to predict final sucess on unseen data. This is a law
            Note: The lock mechanism used to prevent updating champ to a worse testing accuracy(not worse live feed accur) is inside undo dropout 
    We have utility functions to print mosiacs of the data sets for debug. 
        We can load small data sets for debug. To produce fast epochs in development
    The ground work of polymorphic changes to loss functions is in place. As well as polymorphic changes to step size algorithms for mini batch gradient descent
    A logger is enabeled though semi confusing
        Esoteric referes to the small data set you transfer learn onto. Testing is a partion that is itself partioned for the lock. Though with fit to my data on testing is only one partition
        If the data set is too small epoch prints are disabled. De declutter read out. 
    Drop out is properly implemented using best practices
        My initution of dropout is
            Drop out forces the brain to learn duality. Given different variables, find the same solution. For example, a geometric explanation of the pythagorian theorm when coupled with a algebraic expalanation, is a more reslilent thoerm in terms of noise or loss of parameters
    Regularization is added with hyperparameters able to be tuned. This will punish larger weights. 
        Use the regulizer as a case study of how to simply add terms to the loss function. Examine that terms are independent of each other in calculus. Thus the simplicity
        I use a threshold with the regulizer. If the weights are below the threshold. No penality. This avoids adding noise to the gradient with nominal weights. 
    Hyper links are added showing proofs for gradient descent induction
    Numerical stability
        Besides the regulaizer punishing larger weights with exponetial punishment. We use several other, sometimes brilliant methods to achieve stablity 
        Most notably. Examine the link to the proof in softmax
            A method was developed that stabalizes softmax perfectly. Using algebra. Another representation of softmax is found that is equivalent and stable numerically 
        I use simple division to scale input by the number of input parameters. 
        The gradients are normalized. This makes layers with more magnitude have smaller steps 
        Line search algorithm only generates step magnitude none zero if loss is improved. Then larger batch size is used sequentially inorder to minimize pointless steping. Finally the lock is turned on. Making only good steps in terms of testing possible. No matter the numerical processes    
    Filter:
        Static preproccesing of images was used. Not sure how much it helped. There were differences between my handwritting with my tech and the borrowed general data set
        To go further in this direction a CNN should be adopted. Which I will return to one day I hope. 
        
    I reached a satisfactory point. However inorder to properly solve MNIST. A convolutional net is recommended. To maintain a dnn solution the next step is to 
        Add more examples to the my handwritting train directory. Its clear that this will greatly improve accuracy. Examine the read out and add which ones are incorrect the most
            Note that inside various image functions there is a test flag which can be enabled to better trace your operation 

    Next I invert the network inorder to DREAM. Lables in image out. This is a more colorful way of sharing your experience with others. 
    
    Have fun with it. 
    With Love: Palafita

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
        #   The first pass should have the output be around 1/lables for all probablities. Because nothing is known this is expectation. Its a stable numerical start. This is naturally achieved with random uniform numbers 
        
        #   proof of equality https://blester125.com/blog/softmax.html The below line increases numerical stability without changing the value. Fascinating and simple proof. To repeat. this manipulation has not effect on the output. They are equal. Only numerical stability and cost of calculation is added
        dendritic_input -= dendritic_input.max(axis=0) #  out is [0, -inf] This insures that the denominator of softmax never equals 0. 
        e_to_x = np.exp(dendritic_input)
        e_to_x = e_to_x / np.sum(e_to_x, axis=0)
        return e_to_x # todo move return up so no rewrite to e to x in memory optimizationS

    @staticmethod
    def softmax_primed(dendritic_input):
        #   l-1 activation prime used inductively. thus last layers activation primed unneeded
        raise Exception("Back propagation induction allows us to skip the first activations prime")
        #   This is a jacobian
        e_to_x = np.exp(dendritic_input)
        column_sum = np.sum(e_to_x, axis=1)
        coeff = e_to_x / column_sum**2
        return coeff*(column_sum - e_to_x)
        
    @staticmethod
    def none(dendritic_input):
        return dendritic_input

class LOSS_FUNCTIONS:
    accuracy_importance = .9 # [0,1] This may not working properly because induction method differentiates loss in terms of z not activated z
    normality_importance = 1 - accuracy_importance #    This scales the importace of the regulizer
    regulizer_exponential = 12 # Must be even so no negatives 
    regulizer_threshold = 2 #   This prevents the regulizer from punishing nominal weights. Only add punishment if the weights magnitude exceeds the threshold

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
        accuracy_loss = (LOSS_FUNCTIONS.accuracy_importance / supervision.shape[1]) * np.sum(-np.log(dnn.feed_forward(batch)[supervision])) 
        normality_loss = LOSS_FUNCTIONS.normality_importance * LOSS_FUNCTIONS.regularize_weights(dnn)
        return float(accuracy_loss) + float(normality_loss)

    @staticmethod
    def cross_entropy_primed(dnn, batch, supervision):
        #   WARNING: This is the prime of the superivison loss only. The full loss requires other terms primes like regulizer
        #   This is not the complete derivitize. It is the change in loss in terms of z final pre activation = d loss / d activation * d activation / d z =  d loss / d z 
        return (LOSS_FUNCTIONS.accuracy_importance / supervision.shape[1]) * (dnn.feed_forward(batch) - supervision)  

    @staticmethod
    def regularize_weights(dnn):
        #   Punish large weights as theory indicates large weights are indicitive of overfitting. In addition we use drop out to regularize
        punishment = 0
        for layer in dnn.layers: #  This function is called many times with line search where as the primes are called only once per iteraction 
            weights_too_large = np.abs(layer) >= LOSS_FUNCTIONS.regulizer_threshold #   If we punish nominal weights then that will add noise to our gradient. Only punish if over threshold
            punishment += np.sum(layer[weights_too_large]**LOSS_FUNCTIONS.regulizer_exponential)
        return punishment / dnn.network_size

    @staticmethod
    def regularize_weights_primed(dnn):
        #   This is the rate of change of the regulizer term in terms of the weights. Not final z. Seperate from backprogation 
        gradients = [] 
        for layer in dnn.layers:
            gradient = np.zeros_like(layer)
            weights_too_large = np.abs(layer) >= LOSS_FUNCTIONS.regulizer_threshold
            gradient[weights_too_large] = layer[weights_too_large]
            coeff = (LOSS_FUNCTIONS.normality_importance*LOSS_FUNCTIONS.regulizer_exponential/dnn.network_size)
            gradient[weights_too_large] = coeff * (gradient[weights_too_large]**(LOSS_FUNCTIONS.regulizer_exponential-1))
            gradients.append( gradient )  
        
        return gradients 

class DNN:
    def __init__(self, neurons_per_layer, drop_out_per_layer, data, \
            loss_function_and_prime=[LOSS_FUNCTIONS.cross_entropy, LOSS_FUNCTIONS.cross_entropy_primed],\
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
            #   Initialize weights as from uniform distribution between -1, and 1. Then we divide by neural count so that the layers output does not get larger with more neurons
            return np.random.uniform(-1, 1, last_layers_rows*neurons_count).reshape(neurons_count, last_layers_rows) / neurons_count

        def initialize_biases(layers):
            biases = []
            for layer in layers:
                #   We divide by the layers neurons so that biases do not increase the magnitiude of the flow as it moves layer to layer
                #   However note that gradient desent will optimize from here with possibly unstable numerical magnitudes 
                biases.append( np.random.uniform(-1, 1, layer.shape[0]).reshape(layer.shape[0], 1)  / layer.shape[0] )
            return biases

         #  Attach data set to DNN. You can switch data set at any time for transfer learning
        self.data = data

        #   Store meta data
        self.neurons_per_layer, self.drop_out_per_layer = neurons_per_layer, drop_out_per_layer

        #   Dropout prep
        number_of_neurons_to_keep = []
        all_neuron_indicies      = []         
        for layer in range(len(neurons_per_layer)):
            if drop_out_per_layer[layer] > .9 or drop_out_per_layer[layer] < 0: raise Exception("Drop out is irrational")
            drop_not = (1-drop_out_per_layer[layer])
            #   Restructure the network so that drop out does not affect its size. Then after drop out training we scale all layers with no drop out by not dropout. ie 1 dropout.
            neurons_per_layer[layer] = int( neurons_per_layer[layer] / drop_not ) #  exp. .8(8 / .8) = 1     
           #   Buffer the vars used for drop out that are constant 
            neurons_in_layer = neurons_per_layer[layer]
            all_neuron_indicies.append( np.arange(0, neurons_in_layer) ) # List of all indicies of neurons at given layer
            number_of_neurons_to_keep.append( int(neurons_in_layer * drop_not) ) 
        self.all_neuron_indicies = all_neuron_indicies
        self.number_of_neurons_to_keep = number_of_neurons_to_keep #    The last layer always keeps all its neurons thus is not included
        self.kept_weights = None # Used to revert from dropout 

        #   Initialize all layers weights as np objects with the shapes as given from neurons per layer
        input_layer = initialize_layer(neurons_per_layer[0], data.train_data.shape[0]) #  Plus one is for bias
        layers = [input_layer]
        for i, neural_count in enumerate(neurons_per_layer[1:]): #  Hidden layers 
            layers.append(initialize_layer(neural_count, neurons_per_layer[i]))
        #   Init the final layer. It conforms its shape entirely and is not programable
        layers.append(initialize_layer(data.train_supervision.shape[0], neurons_per_layer[-1]))
        self.layers = layers
        
        #   Data storage for layers outputs and input data as first flow, ie output of nothing
        self.flows = [None] * (len(layers)+1)

        #   Store number of weights of network
        self.network_size = np.sum(np.array([layer.size for layer in layers]))

        #   create biases
        self.biases = initialize_biases(layers)

        #   Buffer Network. This holds the shape of the network for dropout reversion
        self.buffered_layers = [np.copy(layer) for layer in layers] 
        self.buffered_biases = [np.copy(bias) for bias in self.biases]

        #   Tie activation functions and their derivities to dnn
        self.hidden_activation, self.hidden_activation_primed = hidden_layers_activation_and_prime
        self.final_activation, self.final_activation_primed = final_activation_and_prime

        #   Tie Loss and its prime
        self.get_loss, self.loss_primed = loss_function_and_prime 

    def feed_forward(self, input, forward_propagating=False):   
        #   Parse independent vars, ie observation into the networks first layer and process activation func if network has hidden layers 
        if forward_propagating:
            self.flows[0] = input #  I think of the input as just another part of the flow. This simplifies induction
            #   Save each output per layer for backpropagating. optimization 
            if len(self.layers) > 1: #  if only 1 layer then the final layer is the only layer.  
                self.flows[1] = self.layers[0] @ input + self.biases[0] #    Backpropagation uses the weighted input unactivated
                flow = self.hidden_activation( self.flows[1] )
            else:
                self.flows[1] = self.layers[0] @ input + self.biases[0]
                flow = self.final_activation(  self.flows[1] ) 
                return flow    

            #   Flow and use activation function for all hiden layers
            for i, layer in enumerate(self.layers[1:-1]):
                self.flows[i+2] = layer @ flow + self.biases[i+1] 
                flow = self.hidden_activation( self.flows[i+2] )
                
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
                flow = self.hidden_activation( layer @ flow  + self.biases[i+1] )
                
            flow = self.final_activation(self.layers[-1] @ flow + self.biases[-1] )
    
        return flow # flow.reshape(len(flow),1) if batch size 1 reshape needed
        
    def get_accuracy(self, batch, batch_supervision):
        #   Only for classification. Take Pass batches from the testing data partition
        batch_size = batch.shape[1]
        return sum(np.argmax(self.feed_forward(batch), axis=0) == np.argmax(batch_supervision, axis=0)) / batch_size

    def get_gradient(self, batch, supervision, normalize=True):
        
        def normalize_tensors(tensors, biases=False, inter_layer_adjustments=None):
            #   We curve and normalize all layers gradients. After this we will be able to uniformly find the step size for all gradients. Where each layer and each weights magnitudes are already scaled differently from this function. Including bias
            if biases:
                for i, t in enumerate(tensors): 
                    t = np.average(t, axis=1)[:, np.newaxis]
                    tensor_magnitude = np.sum(t**2)**.5 * self.layers[i].shape[1] #  We first normalize by batch member, then we divide the bias gradient by the number of terms in its respective neuron, which is why we multiply the denominator
                    #if tensor_magnitude == 0: tensor_magnitude = 1 # Prevent division by 0
                    if tensor_magnitude == 0: tensor_magnitude = 1 # Prevent division by 0
                    #tensors[i] = (inter_layer_adjustments[i] / tensor_magnitude) * t
                    tensors[i] = t / tensor_magnitude
                    #tensors[i] = t
                       
            else:    
                #magnitudes = np.array( [0.0] * len(tensors) )
                #inter_layer_power = 2 # We curve the layers so the layers with higher gradients have less magnitude and vice versa. This way all layers update closer to uniform
                for i, t in enumerate(tensors):
                    tensor_magnitude = np.sum(t**2)**.5
                    #magnitudes[i] = tensor_magnitude 
                    if tensor_magnitude == 0: continue
                    tensors[i] = t / tensor_magnitude #   Normalize
                # return 1
                # if np.any(magnitudes == 0): return np.ones_like(magnitudes) # This is a very bad indicator. Consider putting this condition into debuger condition. Means the gradient is zeroed with some layer. Returning 1 allows continued opperation
                # curved_tensor_magnitudes = magnitudes ** inter_layer_power
                # inter_layer_adjustments = 1 - (curved_tensor_magnitudes / np.sum(curved_tensor_magnitudes)) #   The last layers have more magnitude, thus we must scale their gradients down because they are more sensitive to change, and vice versa. Curve effect with exponet  
                # inter_layer_and_normalization = inter_layer_adjustments / magnitudes
                # for i, layer in enumerate(tensors):
                #     tensors[i] = layer * inter_layer_and_normalization[i] 
                # return inter_layer_adjustments 
                pass

        """
            The idea of backpropagation is that a networks loss from the perspective of a layer is not affected by previous layers. Because a net feeds forward
            Therefore we use the chain rule to only process the loss in terms of the layer being looked at, not earlier layers. Treating all other layers as constant and the ith layer in terms of loss as a function of its weighted input and the the ith+1's gradient. 
            We backprogate because the chain rule goes from the outter functions in, and in until the variable is reached. Ie a specific layer
                delta loss / weights_3 = delta loss / activation final * delta activation final / z final * delta z final / weights_3 where delta z final / weights_3 is expanded with the chain rule till d weights_3/d weights_3 is reached 
                Notice that previous factors of the layers derivitive are allready known from earlier layers derivitives. 
                We start by hand taking the deriviative. 
                I SPENT 4 DAYS WITH THIS PROBLEM. !!!
                    Cross entropy prime and softmax prime merge to form yhat - y.  this is delta cross entropy / a_final * delta a_final / z final = yhat - y
                    Thus the second layer which used the firsts partial does not use relu prime of previous layers activation
                    Also!! THe last layer is the input data. Thus is has no activation. This was the killer. Do no activate last term of delta loss/first layer weights      
            We set the variables of our loss function differently from one point to another, inorder simplify induction and differentiatation
            loss in terms of flows, or loss in terms of weights for a given layer etc. We do not differentiate in terms of the first flow which is the batch, thus the size difference between flows and loss in terms of flows  
            Review these, they gave me trouble 
                https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
                https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-2-693d4162e779
                These explanations above did not work. Note: delta loss / delta flow = delta loss / delta activation * delta activation / delta flow. notice delta activation in denominator and numerator. They cancel 
                The below explanation did work. Above may work now that I understand it better
                https://github.com/marcospgp/backpropagation-from-scratch/blob/master/backpropagation-from-scratch.pdf
                Look at the proof to understand back propagation. This code is designed to run. Its hard to read because of its indexing used for induction
            The induction in back prop takes 2 steps before its recursive for partial loss in terms of weighted inputs. 
            For the loss in tems of the weights using the above partial as a component. The induction is recursive for all the the first and last layers. 
        
        In general, when finding gradients. Use the partial derivitive of a specific weight using rows and columns as indicies, then generalize into vector notation. 
        """
        
        #   TODO optimize by storing activated flows  

        #   Forward Propagate flows 
        self.feed_forward(batch, forward_propagating=True) #    Store the layers outputs to class
        
        #   Backpropagate Loss in terms of flows
        loss_in_terms_of_z_final = self.loss_primed(self, batch, supervision) 
        supervision_loss_in_terms_of_activations = [None] * len(self.layers)
        supervision_loss_in_terms_of_activations[-1] = loss_in_terms_of_z_final # First step of induction is done by hand. 
        supervision_loss_in_terms_of_activations[-2] = self.layers[-1].transpose() @ supervision_loss_in_terms_of_activations[-1] # Second step uses the proof that combines delta cross entropy with delta softmax. They are factors of each other by the chain rule   
        for layer in reversed( range(len(self.layers)-2) ): #    Backpropagation is in reverse order. Final done apove, thats why minus one to length
            supervision_loss_in_terms_of_activations[layer] = self.layers[layer+1].transpose() @ ( self.hidden_activation_primed(self.flows[layer+2]) * supervision_loss_in_terms_of_activations[layer+1] ) # layer + 1 is actually better thought of as the current layer. I saved the first flow as the batch. Flows is 1 more in shape than layers  
        
        #   Loss in terms of a layers weights. Think of this as loss in terms of activations for the layer times its input and output. out_transpose * delta @ in_transpose
        supervision_loss_in_terms_of_weights = [None]*len(self.layers) # Each index is its respective layers partials
        supervision_loss_in_terms_of_weights[0] = (self.hidden_activation_primed(self.flows[1]) * supervision_loss_in_terms_of_activations[0]) @ self.flows[0].transpose() #  First layer is not inductive because the input is not activated. its the data set
        for layer, supervision_loss_in_terms_of_activation in enumerate(supervision_loss_in_terms_of_activations[1:-1]):  
            layer += 1 #    First layer skipped
            supervision_loss_in_terms_of_weights[layer] = (self.hidden_activation_primed(self.flows[layer+1]) * supervision_loss_in_terms_of_activations[layer]) @ self.hidden_activation( self.flows[layer].transpose() ) #  Note: Flows is started with the batch. Thus flows layer is actually the previous layer because flows len is one more than layers    
        supervision_loss_in_terms_of_weights[-1] = supervision_loss_in_terms_of_activations[-1] @ self.hidden_activation( self.flows[-2].transpose() )  

        regulizer_loss_in_term_of_weights = LOSS_FUNCTIONS.regularize_weights_primed(self) # Do not normalize. This is a term of the loss
        total_loss_in_terms_of_weights = [ partial + regulizer_loss_in_term_of_weights[layer] for layer, partial in enumerate( supervision_loss_in_terms_of_weights )] #  Do not weight this sum. This is the proven gradient. Weight the importance of loss's terms in LOSS functions static var for that

        supervision_loss_in_terms_of_biases = [None] * len(self.layers)
        supervision_loss_in_terms_of_biases[-1] = supervision_loss_in_terms_of_activations[-1] #  Not inductive yet because of derivitive of loss and chain to derivitive of final activation 
        for layer, supervision_loss_in_terms_of_flow in enumerate(supervision_loss_in_terms_of_activations[:-1]): #   Last layer already done
            supervision_loss_in_terms_of_biases[layer] = self.hidden_activation_primed(self.flows[layer+1]) * supervision_loss_in_terms_of_activations[layer] #   Loss in terms of bias is just the rate of change of loss in terms of the flow the biase addes to. Because bias just adds to to the flow linearly. Using the chain rule we get to loss in terms of flow from loss in terms of activation which is its outter function. 
        
        if normalize: #   We can now normalize the full gradients in place in terms of each layer.   
            # inter_layer_adjustment = normalize_tensors(total_loss_in_terms_of_weights) #   Early layers have less magnitude, last layer the most. Normalizing will up the first layers and down the last
            # normalize_tensors(supervision_loss_in_terms_of_biases, biases=True, inter_layer_adjustments=inter_layer_adjustment) # Warning: If you place this before loss in terms of flows is used it will break because that changes dependent the input in place
            normalize_tensors(total_loss_in_terms_of_weights)
            normalize_tensors(supervision_loss_in_terms_of_biases, biases=True) # Warning: If you place this before loss in terms of weights is called it will break because that changes dependencies in place

        return total_loss_in_terms_of_weights, supervision_loss_in_terms_of_biases

    def fit(self, batch_size=12, epochs_limit=3, algorithm="parabola", fit_to_my_data=False):

        def delta_loss(step_magnitude, last_loss, buffered_network, buffered_biases):
            #   Only a temp update of the layer
            self.layers = [buffered_network[i] - step_magnitude*layers_gradient for i, layers_gradient in enumerate(layers_gradients)]
            self.biases = [buffered_biases[i]  - step_magnitude*layers_gradient for i, layers_gradient in enumerate(bias_gradients)]
            if use_semi_test:
                perspective_loss = self.get_loss(self, semi_test_batch, semi_test_batch_supervision)                
            else: 
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
            
                This is the linear algebra process to obtain the vertex
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

        def line_search_parabola():
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
            
            if use_semi_test:
                last_loss = self.get_loss(self, semi_test_batch, semi_test_batch_supervision)               
            else:
                last_loss = self.get_loss(self, batch, batch_supervision) #   Compaire each potential step size against loss of no step size in delta loss 
            last_step = step_reset

            buffered_network = list.copy(self.layers) 
            buffered_biases = list.copy(self.biases)
            d_loss, perspective_loss = delta_loss(last_step, last_loss, buffered_network, buffered_biases)
            loop_count = 0
            while (d_loss >= 0): #  While there is no improvement with step length, half the step and check again
                last_step *= .5 #   Half step. Remember, tangent to 0 step is always negative delta loss. So there is always a solution
                d_loss, perspective_loss = delta_loss(last_step, last_loss, buffered_network, buffered_biases)
                loop_count += 1 #   In the event that the network is perfectly fit to the data there will be no improvment possible. Break
                if loop_count == 30:
                    return 0.0, 0.0 #  Futer iterations could still provide improvment but this batch is perfectly fit 
                    
                if d_loss == 0:
                    if np.sum(self.feed_forward(batch)) == 0: 
                        raise Exception("Network is Zeroed out from reLU")
                        #   The reLU function is causing the losses to equal because the output equals zero pre and post descent
                        #   This is appears to be a fatal error. Solution may be to use leaky reLU. if < 0 then x *= -.001  
                
            #   We now know that the delta loss is negative. Lets optimize the step size further with parabula then finalize step
            parabala_points_x = [0, last_step,   last_step*1.5] 
            upstep_delta_loss, upstep_loss = delta_loss(parabala_points_x[2], last_loss ,buffered_network, buffered_biases)
            parabala_points_y = [0, d_loss, upstep_delta_loss] 
            parabala_points_loss = [last_loss, perspective_loss, upstep_loss]

            #   Parabalas need three non linear points. X points will always be different
            if parabala_points_y[1] == parabala_points_y[2]: 
                best_step_index = 1 #   This step is known to be good. Parabula didnt help
            else:
                #   Now using the the three points find the min of a parabala. delta_loss(step_size) only min
                parabala_step = get_parabola_min(parabala_points_x, parabala_points_y)
                #    Parabala vertex can be outside safe step size. If so dont add  
                if parabala_step > step_bounds[0] and parabala_step <= step_bounds[1]: 
                    delta_loss_parabala, parabala_loss = delta_loss(parabala_step, last_loss, buffered_network, buffered_biases)
                    #   Find the min of all steps, then use it as final step. This to insure that steps can only lower loss
                    parabala_points_x.append(parabala_step)
                    parabala_points_y.append(delta_loss_parabala)
                    parabala_points_loss.append(parabala_loss)
                best_step_index = np_.array([parabala_points_y]).argmin()

            if parabala_points_x[best_step_index] == 0: 
                raise Exception("Only negative delta loss should be at this point. Logic bad rewrite")

            #   Finalize Step and prepair for next iteration
            best_known_step = parabala_points_x[best_step_index] 
            last_step = best_known_step
            #last_loss = parabala_points_loss[best_step_index]
            #self.layers = [buffered_network[i] - best_known_step*layers_gradient for i, layers_gradient in enumerate(layers_gradients)]
            self.layers = buffered_network
            self.biases = buffered_biases
            return last_step, last_step#.0001*last_step
            
        def static_steps():
            #    Step a portion of the normal gradient. Note. In a uniform dist, the scale of elements is changed by 1/size. Thus you might vanish. 
            normalize_gradients = True 
            return 1, .001 #    weights step, biases step. Not thought out well

        def read_out(message_type):
            if   message_type == "init":
                if normalize_gradients:
                    normalize_gradients_ = " using Normalized Gradients"
                else: 
                    normalize_gradients_ = " not using Normalized Gradients"
                print("\n\n\n\n\t\t\t\t\t\t\t\t\t\t\t\t\tFIT\n\n" + "\t\t\t\t\t\t\t\t\tStep Algorithm: " + algorithm.__name__.upper() + normalize_gradients_)
                print("\n\t\t\t\tNeurons per Layer: " + str(self.neurons_per_layer) + "\t\t\t Batch Size: " + str(batch_size) + "\t\t\tDropout per Layer: " + str(self.drop_out_per_layer) + \
                    "\t Weights of terms in loss function: Supervision " + str(np.round(LOSS_FUNCTIONS.accuracy_importance,2)) + ", Regulizer " + str(np.round(LOSS_FUNCTIONS.normality_importance,2)) )        
            elif message_type == "epoch":
                print("\n\n\t\t\t\t\t\t\t\t\t\t\t\t\tEPOCH: " + str(epoch+1) + "\n\t\t\t\t-----------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            elif message_type == "progress":
                loss = self.get_loss(self, batch, batch_supervision)
                test_loss = self.get_loss(self, semi_test_batch, semi_test_batch_supervision) 
                if loss < 0.01: 
                    nonlocal use_semi_test
                    if not use_semi_test:
                        use_semi_test = True #    You were able to fit a training batch. Therefor for this batch size/epoch we switch to only taking steps that improve against data not used to build gradients. This effort inorder to counter overfitting to training data while not slowing down training in the beginning
                        print("\n\nUsing Semi test now\n")
                test_accuracy = np.round(self.get_accuracy(test_batch, test_batch_supervision), 2)
                if fit_to_my_data:
                    testing_accuracy_label = "Esoteric Testing Accuracy: "
                else:
                    testing_accuracy_label = "Testing Accuracy: "
                train_accuracy = np.round(self.get_accuracy(batch, batch_supervision), 2)
                print("\t\t\t\t\tIteration: " + str(inter_epoch_iteration) + "\t Step Size: " + f'{step_size_weights:.2E}' + "\t Training Loss: " + str(np.round(loss, 2)) + "\tSemi Testing Loss: " + str(np.round(test_loss, 2))\
                    + "\t Training Accuracy: " + str(train_accuracy) + "\t\t" + testing_accuracy_label + str(test_accuracy))
            elif "final":
                print("\n\n\n\n\n\t\t-------------FINAL----------------\n\nNow with no drop out and the magnitude of the flows scaled by thier dropout rates")
                read_out("progress")
                total_time = (time.time() - start_time) / 60
                print("\nTotal Execution Time: " + f'{total_time:.2E}' + " Minutes")
                self.classify_images()

            else: raise Exception("No message of type")

        def perform_drop_out():
            #   Note: I elected to not scale up the weights after drop out and scale down the weights post reversion. I only scale down at the end of training one time

            def randomize_drop_out():
                for layer in self.all_neuron_indicies:
                    np.random.shuffle(layer)

            randomize_drop_out()

            #   The dropped values are stored in the buffer already from the last iteration: For reversion post iteration
            layers_kept_rows = np.sort( self.all_neuron_indicies[0][:self.number_of_neurons_to_keep[0]] )
            kept_weights = [ layers_kept_rows ]
            #   Drop out neurons. ie rows only because its first layer
            self.layers[0] = self.buffered_layers[0][layers_kept_rows]
            self.biases[0] = self.buffered_biases[0][layers_kept_rows]
            
            #   Dropout for hidden layers has rows and columns from previous layer from matrix multiplication
            for layer, number_of_neurons_to_keep in enumerate(self.number_of_neurons_to_keep[1:]): 
                layers_kept_rows = np.sort( self.all_neuron_indicies[layer+1][:number_of_neurons_to_keep] )
                layers_kept_columns = np.sort( self.all_neuron_indicies[layer][:self.number_of_neurons_to_keep[layer]] )  
                #layers_kept_columns = kept_weights[layer][0]  
                
                kept_weights.append( [ layers_kept_rows, layers_kept_columns] )
                self.layers[layer+1] = self.buffered_layers[layer+1][ layers_kept_rows ][ :, layers_kept_columns ]
                self.biases[layer+1] = self.buffered_biases[layer+1][ layers_kept_rows ]
                

            #   Final layer only drops out columns due to previous layers rows being dropped 
            layers_kept_columns = np.sort( self.all_neuron_indicies[-1][:self.number_of_neurons_to_keep[-1]] )
            kept_weights.append( layers_kept_columns )
            self.layers[-1] = self.buffered_layers[-1][:, layers_kept_columns]
            self.kept_weights = kept_weights
            
        def update_buffers():

            #   add the updated weights to the buffer and reset. First and last layer have different shape 
            self.buffered_layers[0][self.kept_weights[0]] = self.layers[0]
            self.buffered_biases[0][self.kept_weights[0]] = self.biases[0]

            for layer in range(1, len(self.layers)-1):
                # self.buffered_layers[layer][self.kept_weights[layer][0]][:, self.kept_weights[layer][1]] = self.layers[layer]
                #   This is a dirty way of doing the above. The above wont work because it returns a copy due fancy indexing
                rows = np.full(self.buffered_layers[layer].shape, False, dtype=bool)
                columns = np.full(self.buffered_layers[layer].shape, False, dtype=bool)
                rows[self.kept_weights[layer][0]] = True
                columns[:, self.kept_weights[layer][1]] = True
                rows_and_columns = rows * columns
                self.buffered_layers[layer][rows_and_columns] = self.layers[layer].flatten()
                self.buffered_biases[layer][self.kept_weights[layer][0]] = self.biases[layer]
               
            self.buffered_layers[-1][:, self.kept_weights[-1]] = self.layers[-1]
                        
        def undo_drop_out():
            #   Add all dropped out layers back
            self.layers = list.copy( self.buffered_layers )
            self.biases = list.copy( self.buffered_biases )
            
            #   Scale weights to so additional layers dont increase magnitude to activation 
            for layer in range(len(self.layers) - 1):
                restore_total_weight = 1 - self.drop_out_per_layer[layer]
                self.layers[layer] *= restore_total_weight
                self.biases[layer] *= restore_total_weight
            
            # Dont update the champ unless there is improvement to the actual data set your aiming for. Ie if pre transfer dont update
            if use_semi_test: 
                #   If this is the best net yet then store it
                with shelve.open("persistance") as global_storage: #    This will sync storage and close connection at end
                    test_accuracy = self.get_accuracy(semi_test_batch, semi_test_batch_supervision)
                    if test_accuracy > global_storage["best accuracy"]:      
                        improvement_percentage = np.round((test_accuracy/global_storage["best accuracy"] - 1)*100, 2)
                        print("\nA Greater Champion has Emerged !!")
                        print("Previous semi test accuracy was: " + str(np.round(global_storage["best accuracy"], 2)) + " improved to: " + str(np.round(test_accuracy, 2)) + "\n\t" + str(improvement_percentage) + "% Improvment")
                        global_storage["best accuracy"] = test_accuracy
                        data_temp = copy.deepcopy( self.data )
                        self.data.test_data, self.data.train_data = None, None #  We remove the data set from the storage. So that dnn can port way more lightly. We need the de normalize to work though
                        self.data.test_supervision, self.data.train_supervision = None, None 
                        
                        global_storage["champ dnn"] = self
                        self.data = data_temp # Ready up, so dnn can continue training 
                        #global_storage.sync()

            #   Log full nets performance and fitting time
            read_out("final")       
            
            #   Revert to previous state for potential future training 
            for layer in range(len(self.layers) - 1):
                restore_total_weight = 1 - self.drop_out_per_layer[layer]
                self.layers[layer] /= restore_total_weight
                self.biases[layer] /= restore_total_weight

        if fit_to_my_data:
            test_batch, test_batch_supervision = self.data.my_test, self.data.my_test_supervision
        else:
            #   Test used during boot fit. We test against the giant data set before transfer learning on esoteric
            test_sample_size = 500
            test_batch = self.data.test_data[:, 0:test_sample_size]
            test_batch_supervision = self.data.test_supervision[:, 0:test_sample_size]
        


        #   Semi test is used for transfer learning. We insure that any changes help target data set and not just the massive data set used for boot
        # semi_test_batches = self.data.test_data[:, test_sample_size:]
        # semi_test_batches_supervision = self.data.test_supervision[:, test_sample_size:]
        # use_semi_test = False
        # semi_batch_size = 128 # Must be smaller than data. I set this as constant so it never is too large. Because the testing data is smaller than train
        # probability_of_printing_readout_per_iter = 1000 # 1 in 1000 chance of print out
 
        semi_test_batch = self.data.my_test
        semi_test_batch_supervision = self.data.my_test_supervision
        use_semi_test = False
        
        probability_of_printing_readout_per_iter = 1000 # 1 in 1000 chance of print out per epoch
        normalize_gradients = True
        
        #   Parabolic function vars
        step_reset = .6 #    Arbitrary value, no theory. Reseting the Step prevents the step from getting 1.5X bigger potentially each iter. TODO replace multiplication with addition so bonds catching the mid point of parabala and you can remove this reset  
        step_bounds = (0, 1) #    Arbitrary value, no theory. Do not step negatively below lower, or above upper. If parabola vertex interpolates outside bounds then revert to known loss inside bounds
        
        #   Select Step Algorithm
        if algorithm == "parabola": algorithm = line_search_parabola
        else: algorithm = static_steps
        
        read_out("init")
        for epoch in range(epochs_limit): 
            inter_epoch_iteration = 0
            # if batch_size < 1000: # Once you are past 1000 size batch size. Only change weights if they improve on testing data. 
            #     use_semi_test = False
            # else: 
            #     use_semi_test = True
            #     semi_batch_size = 450 # At this point we should just test on the whole semi partition. But I didnt want to recode. Before we reduced size for fast early fitting
            if fit_to_my_data: use_semi_test = True
            if epochs_limit < 3: # keep from congesting terminal
                read_out("epoch") 
            for i in np.random.permutation(np.arange(self.data.train_data.shape[1]-batch_size)): 
                """
                    i = batch iteration
                    Iterate over the entire training data_set every epoch. 
                    The batch size creates the size of the window. From there we increment each the window till the end of the set
                    We optimize the weights every iteration, and we optimize the step size every iteration
                    Use of np.random is explained below
                    We randomize the sequence of iteration. I thought of this personally by this thinking:
                        If the windows are sequential then overfitting will result because all but one of the elements of the last iteration will be the same. 
                        This will cause the net to over fit to that area of the set rather then have a ballanced decent from random windows 
                        Thus we randomize the sequence of windows 
                """
                #   Update batch for current iteration
                inter_epoch_iteration += 1     
                #   We build the gradient from the training data
                batch = self.data.train_data[:, i:i+batch_size]
                batch_supervision = self.data.train_supervision[:, i:i+batch_size]
                
                #   I probe potential steps of the gradient to add to the net with a partition of the testing data. This insures steps always generalize to unseen data and are not overfitting the net to training data 
                #semi_batch_i = np.random.randint(0, semi_test_batches.shape[1] - semi_batch_size)
                #semi_test_batch = semi_test_batches[:, semi_batch_i:semi_batch_i + semi_batch_size]
                #semi_test_batch_supervision = semi_test_batches_supervision[:, semi_batch_i:semi_batch_i + semi_batch_size]                

                #   Drop out neurons randomely to train noise tolerance 
                perform_drop_out()

                #   Update gradient in terms of weights and biases for new iteration's batch
                layers_gradients, bias_gradients = self.get_gradient(batch, batch_supervision, normalize=normalize_gradients) 

                #   Find step size for bias and weights
                step_size_weights, step_size_biases = algorithm()
                if step_size_weights == 0: 
                    # Warning("Check for zeroed out gradients. Feed forward never zeros because of bias")
                    continue # Network could be zerod causes no change in loss with gradient steps of any size. Or the training accuracy is already 100% before gradient step

                #   Perform Mini Batch Gradient Descent
                for layer in range(len(layers_gradients)):
                    self.layers[layer] -= step_size_weights * layers_gradients[layer]
                    self.biases[layer] -= step_size_biases *  bias_gradients[layer]#np.average(bias_gradients[layer], axis=1)[:, np.newaxis]

                #   Update the bufferes with the new trained weights
                update_buffers()

                #   Progress Readout
                if (i % probability_of_printing_readout_per_iter) == 0:                  
                    read_out("progress")

        #   Undo Dropout and Print out final results of fits call, then revert
        undo_drop_out()        

    def classify_images(self, type="filtered"):
        test = False
        #   Put any images you want classified into the live feed folder
        live_feed_dest = "live_feed"
        images = []
        labels = []
        with os.scandir(live_feed_dest) as image_paths:
            for image_path in image_paths:
                color_img = Image.open(live_feed_dest + "/" +  image_path.name)
                gray_img = ImageOps.grayscale(color_img)
                #img = self.data.normalize_tensor( np.array( gray_img ).flatten() ) # img to Cupy => flatten => normalize 
                #gray_img.show( self.data.de_normalize( img).reshape((28,28)) ) #    This tests that the data prep is working by making sure inverse of inverse is the same
                #print(image_path.name)
                img = np.array(gray_img).flatten()
                labels.append(int(image_path.name[0]))
                images.append(img)
        batch = np.zeros(shape=(28**2, len(images))).astype(float) #    Images need to be 28**2
        labels = np.array(labels)
        for i, image in enumerate(images):
            batch[:, i] = image 
        if type == "black or white":
            batch = MNIST.black_and_white(batch, white_and_black=False)
        else:
            batch = MNIST.pre_process_images(batch)
        
        classifications = np.argmax(self.feed_forward(batch), axis=0)
        is_correct = classifications == labels
        not_correct = np.logical_not( is_correct )
        accuracy = str( int(np.round( np.sum( is_correct ) / len(labels) , 2)*100) )
        if test:    # Test that your 
            sampel_size = 3
            print("correct classifications were:   " + str(classifications[is_correct]))
            print("incorrect classifications were: " + str(classifications[not_correct]))
            for i in range(sampel_size):    
                sample = batch[:, not_correct][:, i].reshape(28,28).get()
                active = sample != 0
                sample[active] = 255
                Image.fromarray(sample.astype(np_.uint8), 'L').show() #    L to flag grayscale
        print("\nClassification of live Feed was: \n\tAccuracy was: " + accuracy + "%")
        print( "\nAttempted Classifications for your batch were: " + str(classifications))
        print(   "Correct Classifications would be:              " + str(labels))

class MNIST:

    def __init__(self):
        with shelve.open("persistance") as global_storage:
            self.train_data = global_storage["train primed"]
            self.test_data = global_storage["test primed"]
            self.train_supervision = global_storage["train primed supervision"]
            self.test_supervision = global_storage["test primed supervision"]
            self.my_train = global_storage["my train primed"]
            self.my_train_supervision = global_storage["my train primed supervision"]
            self.my_test = global_storage["my test primed"]
            self.my_test_supervision = global_storage["my test primed supervision"]
            self.live_feed = global_storage["live feed"]
            self.live_feed_supervision = global_storage["live feed supervision"]

    @staticmethod
    def boot(debug=False) -> None:
        #   Load data set from csv => preprocess it => then save it to shelve for fast recall

        def initialize_normalizer():
            active_pixels = self.train_data != 0
            self.std = np.std(self.train_data[active_pixels])
            self.average = np.average(self.train_data[active_pixels])

        def load_data_from_csv(file_location):
            """
                CSV Format: row at image. row 1 as header. Column one as solution lable. 
            """
            #   
            data = np.genfromtxt(file_location, delimiter=',', dtype=np.uint8)[1:] #    First Row is Gumbo     
            
            #   Solution as a one hot vector. In MNIST we have 0-9 as labels. We put the position of the output neurons as the value of the label
            supervision = np.zeros(shape=(10, len(data))) # Ten Labels X Data set rows 
            supervision[data[:,0], range(len(data))] = 1

            #   A instance is a column as input in matrix multiplication thus the transpose. 1: Remove the labels column. labels for each arg from excel etc
            data = data[:,1:].transpose() 
            #data = normalize_tensor(data) # This will convert to z scores from data 
            
            return data, supervision

        def load_data_from_images(file_location, type="filter"):
            test = False
            #   Put any images you want classified into the live feed folder
            images = []
            labels = []
            with os.scandir(file_location) as image_paths:
                for image_path in image_paths:
                    color_img = Image.open(file_location + "/" +  image_path.name)
                    gray_img = ImageOps.grayscale(color_img)
                    #img = self.data.normalize_tensor( np.array( gray_img ).flatten() ) # img to Cupy => flatten => normalize 
                    #gray_img.show( self.data.de_normalize( img).reshape((28,28)) ) #    This tests that the data prep is working by making sure inverse of inverse is the same
                    #print(image_path.name)
                    img = np.array(gray_img).flatten()
                    labels.append(int(image_path.name[0]))
                    images.append(img)
            batch = np.zeros(shape=(28**2, len(images))).astype(float) #    Images need to be 28**2
            labels = np.array(labels)
            labels_ = np.zeros(shape=(10, labels.shape[0]))
            labels_[labels, np.arange(0, len(labels))] = 1
            for i, image in enumerate(images):
                batch[:, i] = image 

            shuffle_i = np.random.permutation( batch.shape[1] )
            batch = batch.transpose()[shuffle_i].transpose()
            labels_ = labels_.transpose()[shuffle_i].transpose()
            if test:    # Test that your doing stuff right by looking at it
                sampel_size = 3
                for i in range(sampel_size):    
                    sample = batch[:, i].reshape(28,28).get()
                    active = sample != 0
                    sample[active] = 255
                    Image.fromarray(sample.astype(np_.uint8), 'L').show() #    L to flag grayscale
                    print("label " + str( np.argmax(labels_, axis=0)[i] ) )
            return batch, labels_

        run_test = False #  Test that normalize and de normalize are inverse. dev time only
        #   Uncomment each section for effect. 

        #   Full train Full test. not recommended
        #self.train_data, self.train_supervision = load_data_from_csv('data_sets/mnist_train.csv')
        #self.test_data, self.test_supervision =   load_data_from_csv('data_sets/mnist_test.csv')

        #   Use this for quick load for development
        if debug:
            train_data, train_supervision = load_data_from_csv('data_sets/mnist_test.csv')
            test_data, test_supervision = train_data[:,9000:], train_supervision[:,9000:] 
            train_data, train_supervision = train_data[:,:9000], train_supervision[:,:9000]
        else:
            #   Full train and sufficient test. Recommended 
            train_data, train_supervision = load_data_from_csv('data_sets/mnist_train.csv')
            test_data, test_supervision =   load_data_from_csv('data_sets/mnist_test.csv')
            test_data, test_supervision = test_data[:,:1000], test_supervision[:,:1000] # This is to speed up operation. I only need 500 sample size for test 
        MNIST.pre_process_images(train_data, "train primed", train_supervision), MNIST.pre_process_images(test_data, "test primed", test_supervision)
        
        #   I transfer learn over to my own handwritting because even with 97 accuracy the net cant figure out change of pen size spacing and style 
        my_handwritting_train, my_handwritting_train_supervision = load_data_from_images("data_sets/my handwritting train")
        my_handwritting_test, my_handwritting_test_supervision =   load_data_from_images("data_sets/my handwritting test")
        live_feed_batch, live_feed_labels =                        load_data_from_images("live_feed")
        MNIST.pre_process_images(live_feed_batch,       "live feed",        live_feed_labels)
        MNIST.pre_process_images(my_handwritting_train, "my train primed",  my_handwritting_train_supervision)
        MNIST.pre_process_images(my_handwritting_test,  "my test primed",   my_handwritting_test_supervision)
        
        #initialize_normalizer()
        # if run_test:
        #     temp = self.test_data
        #     temp_n = self.normalize_tensor(temp)
        #     if np.all( temp == self.de_normalize_tensor(temp_n) ): raise Exception("De normalize and normalize are not inverse")
        #self.train_data, self.test_data = self.normalize_tensor(self.train_data), self.normalize_tensor(self.test_data)
        #self.train_data, self.test_data = MNIST.black_and_white(self.train_data), MNIST.black_and_white(self.test_data)
        pass

    def show_elements(self, data_set):
        #   This function allows you to view random elements of the data set 
        print_labels, compress = False, False
        images_per_row = 30
        
        if data_set == "train":
            data_set = self.train_data
        elif data_set == "test":
            data_set = self.test_data
        elif data_set == "my train":
            data_set = self.my_train
        elif data_set == "my test":
            data_set = self.my_test
        elif data_set == "live feed":
            data_set = self.live_feed 
        
        mosaic_width = (28*images_per_row)
        total_images_to_show = min(images_per_row**2, data_set.shape[1])
        if total_images_to_show > data_set.size: # TODO sort the sample no matter its size so its easier to read 
            elements = np.random.randint(0, data_set.shape[1], total_images_to_show)
        else:
            elements = np.arange(0, data_set.shape[1]) #   Show all in sequential order not random  
        if len(elements) > mosaic_width**2: raise Exception("Too many elments to show")
        mosaic = np_.zeros(shape=(mosaic_width, mosaic_width), dtype=np.uint8)

        for i, element in enumerate(elements):
            # #   Image will not work unless dtype is uint8
            #image = self.de_normalize_tensor( self.test_data[:,element] ).reshape(28, 28).get().astype(np.uint8) #   Denormalize => reshape => cupy to numpy => data type to accepted pixel   
            image = (data_set[:,element] * data_set.shape[0] * 255).reshape(28, 28).get().astype(np.uint8) # for black and white. uncomment above if data is normalized
            row, column = 28*(i // images_per_row), 28*(i % images_per_row)
            mosaic[row:row+28, column:column+28] = image
    
        #if print_labels:    print("\nLables for images: " + str( np.argmax(self.test_supervision[:,elements], axis=0) ) ) # Extraneous 
        if compress: 
            #   If you want to preview some level of compression call this line
            conserve = .25
            Image.fromarray(mosaic, 'L').resize((int(mosaic_width*conserve),int(mosaic_width*conserve))).show()
        else: 
            #Image.fromarray(image, "L").show() #    Use this to test the framing of individual elements
            Image.fromarray(mosaic, 'L').show() #    L to flag grayscale
            
    def normalize_tensor(self, tensor):
        #   For any shape tensor: Normalize the inpute to keep it close to activation value 0. We change data structure to float
        active_pixels = tensor != 0
        #z_scores_of_pixels = (tensor[active_pixels] - np.average(tensor[active_pixels])) / np.std(tensor[active_pixels])
        z_scores_of_pixels = (tensor[active_pixels] - self.average) / self.std
        tensor = np.zeros_like(tensor, dtype=float) #   This changes the data structure from uint8 to float
        tensor[active_pixels] = z_scores_of_pixels
        return tensor / tensor.shape[0] #   This last step I added so that as we use more inputs, the scale gets exponetially smaller, thus keeping the flows closer to 0 where we want them

    def de_normalize_tensor(self, tensor):
        active_pixels = tensor != 0
        tensor[active_pixels] = ((tensor[active_pixels] * tensor.shape[0] * self.std) + self.average).astype(np.uint8)
        return tensor

    @staticmethod
    def black_and_white(tensor, white_and_black=True):
        if white_and_black:
            #   Black is active. We switch it so white is active and scale for stability. 
            tensor_ = np.zeros_like(tensor, dtype=float)
            active = tensor != 255
            tensor_[active] = 1 / tensor.shape[0]
            tensor_[np.logical_not(active)] = 0
            return tensor_
        else:
            tensor_ = np.zeros_like(tensor, dtype=float)
            active = tensor != 0
            tensor_[active] = 1 / tensor.shape[0]
            return tensor_

    @staticmethod
    def to_on_or_off(tensor):
        tensor[tensor != 0] = 255
        return tensor

    @staticmethod
    def pre_process_images(batch, save_to=None, supervison=None):
        #   If a row or column is dead then crop it. Then expand size of image back to its original shape. 
        batch.reshape((batch.shape[1], 28, 28))
        for i, image in enumerate( batch.transpose() ):
            image = image.reshape(28,28)
            
            image[image != 0] = 255 # Black or white
            img = Image.fromarray( image.get().astype(np.uint8), "L" ).resize((30,30))
            img = img.filter(ImageFilter.CONTOUR) # Filter will leave only the contour 
            image = np.array( img )

            active_rows, active_columns = np.sum(image, axis=1) != 0, np.sum(image, axis=0) != 0 #  Remove the border created by contour filter
            image = image[active_rows]
            image = image[:, active_columns]    
            active_rows, active_columns = np.sum(image, axis=1) != 255*image.shape[0], np.sum(image, axis=0) != 255*image.shape[1] #  center image
            image = image[active_rows]
            image = image[:, active_columns]   

            image[image != 255] = 0 #   Remove noise. black or white. no gray
            img = Image.fromarray(image.get().astype(np_.uint8), 'L').resize((28,28)) # Restore original shape
            image = np.array(img)
            image[image != 255] = 0 #   Remove noise. black or white. no gray
            #img.show()

            batch[:, i] = image.flatten()


        batch = MNIST.black_and_white(batch) # swap to white on black off. Also scale the values for numerical stability 
        # Image.fromarray( MNIST.to_on_or_off( batch[:,0]).reshape((28,28)).get().astype(np_.uint8), 'L').show() # test image
        if save_to == None:
            return batch
        with shelve.open("persistance") as global_storage:  
            global_storage[save_to] = batch #save
            global_storage[save_to + " supervision"] = supervison #save
            
        return batch


try_for_better_dnn = False

if not try_for_better_dnn:
    data = MNIST()
    #data.show_elements("train")
    #data.show_elements("test")
    #data.show_elements("my train") 
    #data.show_elements("my test") 
    data.show_elements("live feed") 
    with shelve.open("persistance") as global_storage:
        champ_dnn = global_storage["champ dnn"]

    champ_dnn.classify_images()

else:
    step_algorithm = "parabola"
    start_time = time.time() #  Global start time will allow global access
    
    def boot():
        #   Boot code
        with shelve.open("persistance") as global_storage:
            global_storage["best accuracy"] = 0 #   Use the first time you run. Or any time you want to restart all     
        MNIST.boot(debug=False) #   Debug to true will speed up development by using smaller datasets, now that we load data from shelve its way faster than loading from csv, so no need to use

    def change_data_set(esoteric=True):
        #   Esoteric referes to transfer learning onto a small data set. ie finetunning 
        if esoteric:
            dnn_spawn.data.train_data = data.my_train
            dnn_spawn.data.train_supervision = data.my_train_supervision
            dnn_spawn.data.test_data = data.my_test
            dnn_spawn.data.test_supervision = data.my_test_supervision    
        else:
            dnn_spawn.data.train_data = data.train_data
            dnn_spawn.data.train_supervision = data.train_supervision
            dnn_spawn.data.test_data = data.test_data
            dnn_spawn.data.test_supervision = data.test_supervision

    #boot() #    Only call to erase champ dnn and reload data from images and csv
    hone_champion = True
    data = MNIST() #    Load data for supervised learning of spawns
    
    if hone_champion:
        with shelve.open("persistance") as global_storage:
            dnn_spawn = global_storage["champ dnn"] # Load best known net
            dnn_spawn.data = copy.deepcopy( data )  #   We do not persist the nets data set with it. This is so when we port trained ai it will be light. Here we load its data set so that it can be honed
    else:  
        #   Without honing we create a new net  
        neurons_per_layer = [420, 90, 20] # Logic implicitly solves the columns of a net. Here we specify rows of each layer except the final layer. Rows are neurons count. Last layer rows are infered from data sets supervision  
        drop_out_per_layer = [0.8, .6, 0.5] # Dropout will adapt the net to noise. Missing respective input forces generalization and hardyness
        dnn_spawn = DNN(neurons_per_layer, drop_out_per_layer, data=copy.deepcopy(data))
        #   Launch random net => train on general data to get ball park 
        dnn_spawn.fit(batch_size=32, epochs_limit=1, algorithm=step_algorithm)    
        change_data_set(esoteric=True)
        #   Now we switch to esoteric data without the lock. NOTE The fit_to_my_data lock insures that all changes improve the test data. Gradient from train, checks if it improves test before moving. Lock locks off of testing loss not testing accuracy
        dnn_spawn.fit(batch_size=32, epochs_limit=10, algorithm=step_algorithm, fit_to_my_data=False)
        change_data_set(esoteric=False)
        dnn_spawn.fit(batch_size=128, epochs_limit=1, algorithm=step_algorithm, fit_to_my_data=True)
        dnn_spawn.fit(batch_size=256, epochs_limit=1, algorithm=step_algorithm, fit_to_my_data=True)
    
    #   Now we engage lock and hone. Only good changes possible. With lock on model will be saved if better found
    #   Switch to giant free data set but put in lock . The lock flag is labeled fit_to_my_data
    dnn_spawn.fit(batch_size=32, epochs_limit=1, algorithm=step_algorithm, fit_to_my_data=True)
    
    change_data_set(esoteric=True)
    #   Lastly we hone the model with esoteric data
    dnn_spawn.fit(batch_size=32, epochs_limit=20, algorithm=step_algorithm, fit_to_my_data=True)



"""
    Change so you can store multiple nets and load different nets 
    solve mnist with mse
    make net polymorphic so you can change input and output size 
    solve for mnist inverted 


        Goals

    train on labels to images then print 
    
    feed generator to classifierer and test accur 

    Include a classification neuron for unclassified so that the net is not forced to classify stuff it cannot recognize

    add numerical gradients as in video for test.  This will test if your gradient is correct
        https://www.youtube.com/watch?v=pHMzNW8Agq4&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU&index=5
     



   # Mean squared error method
    loss_primed = self.loss_primed(self, batch, supervision) 
    self.feed_forward(batch, forward_propagating=True) #    Store the layers outputs to class
    gradient_layer_1 = self.layers[1].transpose() @ loss_primed @ self.hidden_activation_primed( batch.transpose() ) 
    gradient_layer_2 = loss_primed @ self.hidden_activation(self.flows[0].transpose())

    dnn = DNN(neurons_per_layer, data=data, final_activation_and_prime=[ACTIVATIONS.none, ACTIVATIONS.none], loss_function_and_prime=[LOSS_FUNCTIONS.mean_squared_error, LOSS_FUNCTIONS.mean_squared_error_primed])

"""