module Main where

import Control.Monad
import System.Random
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Neural

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
