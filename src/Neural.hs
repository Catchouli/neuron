module Neural
  ( Layer(Layer)
  , Network
  , sigmoid
  , sigmoid'
  , evalLayer
  , evalLayerM
  , evalNetwork
  , evalNetworkM
  , evalOutputError
  , evalOutputErrorM
  , evalLayerError
  , evalLayerErrorM
  , evalNetworkError
  , evalNetworkErrorM
  , gradientDescent
  , gradientDescentM
  , sumColumns
  )
where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data

-- Dependent types would be great to represent some of these constraints.

-- A layer in the neural network. The weights are represented by a matrix with
-- 1 row for each neuron, and 1 column for each weight. The biases are represented
-- by a vector. The number of biases must match the number of weights.
data Layer = Layer (Matrix R) (Vector R) deriving Show

-- A network is a list of layers, where each layer has the same number of weights as
-- the previous one has neurons. Otherwise, evalNetwork will fail.
type Network = [Layer]

-- A utility function that converts a vector to a row matrix
vecToMat :: Vector R -> Matrix R
vecToMat = fromRows . (:[])

-- A utility function that sums the colums in a matrix to produce a row vector
sumColumns :: Matrix R -> Vector R
sumColumns m = (vector . replicate (rows m) $ 1) <# m

-- A sigmoid function which is at 0.5 for the input 0, and tends to 1 as the input
-- increases, and to 0 as the input decreases.
sigmoid :: Floating f => f -> f
sigmoid z = 1 / (1 + exp (-z))

-- Differential of the sigmoid fuunction
sigmoid' :: Floating f => f -> f
sigmoid' z = sigmoid z * (1 - sigmoid z) 

-- Evaluates a layer of the neural network given a set of inputs. The number of inputs
-- must match the number of weights in the input network or this function will fail
-- with an exception. Returns the outputs as well as the weighted inputs.
evalLayer :: Layer -> Vector R -> (Vector R, Vector R)
evalLayer layer inputs = (flatten output, flatten weightedInput)
  where
    (output, weightedInput) = evalLayerM layer (fromRows [inputs])

-- Evaluates a layer of the neural network given a set of inputs. Contrary to evalLayer,
-- the set of inputs is provided as a matrix where each row is a vector of inputs as
-- given in evalLayer.
evalLayerM :: Layer -> Matrix R -> (Matrix R, Matrix R)
evalLayerM (Layer weights biases) inputs = (sigmoid weightedInputs, weightedInputs)
  where
    inputRows = rows inputs
    biasMatrix = fromColumns . replicate inputRows $ biases
    --weightedInputs = tr' $ (weights <> tr' inputs) + biasMatrix
    weightedInputs = tr' $ (weights <> tr' inputs) + biasMatrix

-- Feed forward the given function (e.g. evalLayer) over the network
feedForward :: (Layer -> a -> (a, a)) -> a -> Network -> [(a, a)]
feedForward f inputs network = tail . reverse $ feedForward' f inputs network
  where
    feedForward' f inputs network = foldl (\acc@((a,_):_) l -> (f l a) : acc) [(inputs, undefined)] network

-- Evaluates a neural network by folding a set of inputs through each layer.
-- The number of inputs must match the number of weights in the first layer,
-- and the number of elements in the output vector depends on the number of neurons
-- in the final layer of the network. Returns the output and weighted input of each
-- layer in the network.
evalNetwork :: Network -> Vector R -> [(Vector R, Vector R)]
evalNetwork network inputs = feedForward evalLayer inputs network

-- Evaluates a neural network. Contrary to evalNetwork, the input is taken in the same form
-- as evalLayerM, i.e. a matrix where the rows are a single input vector.
evalNetworkM :: Network -> Matrix R -> [(Matrix R, Matrix R)]
evalNetworkM network inputs = feedForward evalLayerM inputs network

-- Evaluates the error in the output layer of a network
evalOutputError :: Vector R -> Vector R -> Vector R -> Vector R -> Vector R
evalOutputError output weightedInput inputs training = (output - training) * sigmoid' weightedInput

-- Evaluates the error in the output layer of a network. Input is a matrix where rows are inputs.
evalOutputErrorM :: Matrix R -> Matrix R -> Matrix R -> Matrix R -> Matrix R
evalOutputErrorM output weightedInput inputs training = (output - training) * sigmoid' weightedInput

-- Evaluates the error in a layer of a network given the weighted cost, and the next layer's weight
-- and error.
evalLayerError :: Vector R -> Matrix R -> Vector R -> Vector R
evalLayerError weightedCost nextLayerWeights nextLayerError =
  ((tr' nextLayerWeights) #> nextLayerError) * (sigmoid' weightedCost)

-- Evaluates the error in a layer like evalLayerError, but takes a matrix of inputs as rows.
evalLayerErrorM :: Matrix R -> Matrix R -> Matrix R -> Matrix R
evalLayerErrorM weightedCost nextLayerWeights nextLayerError =
  (nextLayerError <> nextLayerWeights) * (sigmoid' weightedCost)

-- Evaluates the error for each layer in the network by means of backpropogation.
evalNetworkError :: Network -> [(Vector R, Vector R)] -> Vector R -> Vector R -> [Vector R]
evalNetworkError network networkValues inputs training =
  map flatten $ evalNetworkErrorM network networkValuesM (vecToMat inputs) (vecToMat training)
    where
      networkValuesM = map (\(a,b) -> (vecToMat a, vecToMat b)) networkValues

-- Evaluates the error for each layer in the network by means of backpropogation.
evalNetworkErrorM :: Network -> [(Matrix R, Matrix R)] -> Matrix R -> Matrix R -> [Matrix R]
evalNetworkErrorM network networkValues inputs training = map fst backPropogation
  where
    -- The layers of the network in reverse order
    reverseNetwork = reverse network
    -- The activations of the layers of the network in reverse order
    reverseLayerVals = reverse networkValues
    -- (reverse layers, reverse activations)
    reverseLayerData = zip reverseNetwork reverseLayerVals
    -- The activations of the output layer of the network
    outputLayer = tail reverseLayerVals
    -- The weights of the output layer
    Layer outputWeights _ = head reverseNetwork
    -- The activations and weighted inputs of the output layer
    (outputValues, outputLayerWeightedInput) = head reverseLayerVals
    -- The error of the output layer
    outputError = evalOutputErrorM outputValues outputLayerWeightedInput inputs training
    -- An accumulator for backpropogation, started with the output layer's error
    acc = [(outputError, outputWeights)]
    -- The backpropogation algorithhm (apply evalOutputError to each layer in reverse order)
    backPropogate = \acc@((loe,lw):_) (Layer w _, (_,z)) -> (evalLayerErrorM z lw loe, w):acc
    backPropogation = foldl backPropogate acc (tail reverseLayerData)

-- Gradient descent function. Given a network, its inputs, its activations, and its error,
-- produces a new network corrected through gradient descent.
gradientDescent :: Network -> Vector R -> [Vector R] -> [Vector R] -> R -> Network
gradientDescent network input activations error learnRate =
  gradientDescentM network (vecToMat input) (map vecToMat activations) (map vecToMat error) learnRate

-- Gradient descent function. Given a network, its inputs, its activations, and its error,
-- produces a new network corrected through gradient descent.
gradientDescentM :: Network -> Matrix R -> [Matrix R] -> [Matrix R] -> R -> Network
gradientDescentM network input activations error learnRate =
  map gradientDescent' layerData
    where
      -- Get the activations from the previous layer of neurons
      -- by offsetting the activations with the input at the start
      -- and discarding the final element
      previousActivations = input : (init activations)
      -- Layers zipped with their errors and the previous layer's activation
      layerData = zip3 network error previousActivations
      -- learnRate as fractional
      learnRateF = realToFrac learnRate
      -- Gradient descent for one layer
      gradientDescent' (Layer weight bias, error, previousActivation) =
        -- New layer is evaluated by calculating the gradient for the weight and bias and
        -- subtracting gradient * learnRate from the current value
        Layer
          (weight - (weightGradients * realToFrac learnRate))
          (bias - (biasGradients * realToFrac learnRate))
            where
              -- Calculate gradient for each rate
              -- Calculated using the kronecker product, a generalisation of tensor product to matrices
              weightGradients = kronecker previousActivation (tr' error)
              -- Sum the bias gradient components for each neuron (the columns)
              biasGradients = sumColumns error
