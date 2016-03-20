{-# LANGUAGE OverloadedStrings #-}

module Main where

import Prelude hiding (readFile, drop)
import Control.Monad
import Control.Monad.IO.Class
import System.Random
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import SDL
import Debug.Trace
import qualified Linear

import Neural
import MNIST

-- Generates a layer with a given number of neurons with inputCount weights.
-- Weights are randomised between 0 and 1.
genLayer :: Int -> Int -> IO Layer
genLayer neurons inputCount = do
  weights <- replicateM (neurons * inputCount) (randomRIO (-0.2, 0.2) :: IO R)
  biases <- replicateM (neurons) (randomRIO (-0.2, 0.2) :: IO R)
  return $ Layer (matrix inputCount weights) (vector biases)

-- Generates a network with the specified number of neurons in each layer,
-- and the number of inputs to the network.
-- Initialised with random7004 weights and biases as in genLayer.
genNetwork :: [Int] -> Int -> IO Network
genNetwork layers inputCount = sequence . map (uncurry genLayer) $ layerDefs
  where
    -- Layer definitions in the form (neuronCount, inputCount)
    layerDefs = zip layers (inputCount:layers)

-- A test network
testNetwork :: IO Network
testNetwork = do
  l1 <- genLayer 5 3
  l2 <- genLayer 5 5
  l3 <- genLayer 1 5
  return [l1, l2, l3]

-- Another test network
testNetwork2 :: Network
testNetwork2 = [hiddenLayer, outputLayer]
  where
    hiddenLayer = Layer (matrix 2 [0.15, 0.2, 0.25, 0.3]) (vector [0.35, 0.35])
    outputLayer = Layer (matrix 2 [0.4, 0.45, 0.5, 0.55]) (vector [0.6, 0.6])

-- Test inputs for testNetwork2
testInputs = vector [0.05, 0.1]

-- Training data for testInputs
trainingData = vector [0.01, 0.99]

-- Train neural network
trainNet = do
  (imageCount, width, height, imageData) <- readMNISTImages "train-images-idx3-ubyte"
  (labelCount, labels) <- readMNISTLabels "train-labels-idx1-ubyte"

  let image = head imageData
  let input = vector (map fromIntegral image)
  let correctNumber = fromIntegral . head $ labels
  let training = vector $ map (\i -> if i == correctNumber then 1.0 else 0.0) [1..10]

  hiddenLayer <- genLayer 20 (width * height)
  outputLayer <- genLayer 10 20

  let initialNetwork = [hiddenLayer, outputLayer]

  let loop net i err =
        case i > 1000 || err < 0.1 of
          True  -> net
          False -> trace (show $ sumElements (last error)) $ loop nextNet (i+1) (sumElements (last error))
            where
              output = evalNetwork net input
              error = evalNetworkError net output input training
              nextNet = gradientDescent net input (map fst output) error 0.1

  let trainedNet = loop initialNetwork 0 1000

  return (input, initialNetwork, trainedNet, training)

-- Train neural network
trainNetM = do
  (imageCount, width, height, imageData) <- readMNISTImages "train-images-idx3-ubyte"
  (labelCount, labels) <- readMNISTLabels "train-labels-idx1-ubyte"

  let image = head imageData
  let input = fromRows [vector (map fromIntegral image)]
  let correctNumber = fromIntegral . head $ labels
  let training = fromRows [vector $ map (\i -> if i == correctNumber then 1.0 else 0.0) [1..10]]

  hiddenLayer <- genLayer 20 (width * height)
  outputLayer <- genLayer 10 20

  let initialNetwork = [hiddenLayer, outputLayer]

  let loop net i err =
        case i > 100000 || err < 0.1 of
          True  -> net
          False -> trace (show $ sumElements (last error)) $ loop nextNet (i+1) (sumElements (last error))
            where
              output = evalNetworkM net input
              error = evalNetworkErrorM net output input training
              nextNet = gradientDescentM net input (map fst output) error 0.1

  let trainedNet = loop initialNetwork 0 1000

  return (input, initialNetwork, trainedNet, training)

-- Make a window and show MNIST image
window = do
  initializeAll

  window <- createWindow "SDL Application" defaultWindow
  renderer <- createRenderer window (-1) defaultRenderer

  -- Read in mnist data
  (imageCount, width, height, imageData) <- readMNISTImages "train-images-idx3-ubyte"
  (labelCount, labels) <- readMNISTLabels "train-labels-idx1-ubyte"

  -- Convert to sdl texture
  texture <- loadTextureMNIST renderer width height $ head imageData

  SDL.rendererDrawColor renderer $= Linear.V4 255 255 255 255
  SDL.clear renderer

  copy renderer texture Nothing Nothing

  present renderer

  let loop = do
      pollEvents
  
      keyState <- getKeyboardState

      case keyState ScancodeEscape of
        True -> (return ()) :: IO ()
        False -> loop

  loop

  destroyTexture texture
  destroyRenderer renderer
  destroyWindow window
