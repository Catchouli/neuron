module Main where

import Control.Exception
import Control.Monad
import Data.Traversable
import System.Random
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
evalLayer (Layer weights biases) inputs = (sigmoid weightedInputs, weightedInputs)
  where
    weightedInputs = (weights #> inputs) + biases

-- Evaluates a neural network by folding a set of inputs through each layer.
-- The number of inputs must match the number of weights in the first layer,
-- and the number of elements in the output vector depends on the number of neurons
-- in the final layer of the network. Returns the output and weighted input of each
-- layer in the network.
evalNetwork :: Network -> Vector R -> [(Vector R, Vector R)]
evalNetwork network inputs = tail . reverse $ feedForward inputs network
  where
    feedForward inputs network = foldl (\acc@((a,_):_) l -> (evalLayer l a) : acc) [(inputs, vector [])] network

-- Evaluates the error in the output layer of a network
evalOutputError :: Vector R -> Vector R -> Vector R -> Vector R -> Vector R
evalOutputError output weightedInput inputs training = (output - training) * sigmoid' weightedInput

-- Evaluates the error in a layer of a network given the weighted cost, and the next layer's weight
-- and error.
evalLayerError :: Vector R -> Matrix R -> Vector R -> Vector R
evalLayerError weightedCost nextLayerWeights nextLayerError =
  ((tr' nextLayerWeights) #> nextLayerError) * sigmoid' weightedCost

-- Evaluates the error for each layer in the network by means of backpropogation.
evalNetworkError :: Network -> [(Vector R, Vector R)] -> Vector R -> Vector R -> [Vector R]
evalNetworkError network networkValues inputs training = map fst backPropogation
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
    outputError = evalOutputError outputValues outputLayerWeightedInput inputs training
    -- An accumulator for backpropogation, started with the output layer's error
    acc = [(outputError, outputWeights)]
    -- The backpropogation algorithhm (apply evalOutputError to each layer in reverse order)
    backPropogate = \acc@((loe,lw):_) (Layer w _, (_,z)) -> (evalLayerError z lw loe, w):acc
    backPropogation = foldl backPropogate acc (tail reverseLayerData)

-- Generates a layer with a given number of neurons with inputCount weights.
-- Weights are randomised between 0 and 1.
genLayer :: Int -> Int -> IO Layer
genLayer neurons inputCount = do
  weights <- replicateM (neurons * inputCount) (randomRIO (0.0, 1.0) :: IO R)
  biases <- replicateM (neurons) (randomRIO (0.0, 1.0) :: IO R)
  return $ Layer (matrix inputCount weights) (vector biases)

-- Generates a network with the specified number of neurons in each layer,
-- and the number of inputs to the network.
-- Initialised with random weights and biases as in genLayer.
genNetwork :: [Int] -> Int -> IO Network
genNetwork layers inputCount = sequence . map (uncurry genLayer) $ layerDefs
  where
    -- Layer definitions in the form (neuronCount, inputCount)
    layerDefs = zip layers (inputCount:layers)

-- Gradient descent function. Given a network, its inputs, its activations, and its error,
-- produces a new network corrected through gradient descent.
gradientDescent :: Network -> Vector R -> [Vector R] -> [Vector R] -> R -> Network
gradientDescent network input activations error learnRate =
  map gradientDescent' layerData
    where
      -- Get the activations from the previous layer of neurons
      -- by offsetting the activations with the input at the start
      -- and discarding the final element
      previousActivations = input : (init activations)
      -- Layers zipped with their errors and the previous layer's activation
      layerData = zip3 network error previousActivations
      -- Gradient descent for one layer
      gradientDescent' (Layer weight bias, error, previousActivation) =
        Layer (weight - weightGradient) (bias - biasGradient)
          where
            weightGradient = outer error previousActivation / (realToFrac learnRate)
            biasGradient = error / (realToFrac learnRate)

testNetwork :: IO Network
testNetwork = do
  l1 <- genLayer 5 3
  l2 <- genLayer 5 5
  l3 <- genLayer 1 5
  return [l1, l2, l3]

testNetwork2 :: Network
testNetwork2 = [hiddenLayer, outputLayer]
  where
    hiddenLayer = Layer (matrix 2 [0.15, 0.2, 0.25, 0.3]) (vector [0.35, 0.35])
    outputLayer = Layer (matrix 2 [0.4, 0.45, 0.5, 0.55]) (vector [0.6, 0.6])

testInputs = vector [0.05, 0.1]

trainingData = vector [0.01, 0.99]
