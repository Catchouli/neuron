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

-- Evaluates a layer of the neural network given a set of inputs. The number of inputs
-- must match the number of weights in the input network or this function will fail
-- with an exception.
evalLayer :: Layer -> Vector R -> Vector R
evalLayer (Layer weights biases) inputs = sigmoid ((weights #> inputs) + biases)

-- Evaluates a neural network by folding a set of inputs through each layer.
-- The number of inputs must match the number of weights in the first layer,
-- and the number of elements in the output vector depends on the number of neurons
-- in the final layer of the network.
evalNetwork :: Network -> Vector R -> Vector R
evalNetwork layers inputs = foldl (flip evalLayer) inputs layers

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
