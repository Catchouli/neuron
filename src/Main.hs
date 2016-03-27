{-# LANGUAGE OverloadedStrings #-}

module Main where

import Prelude hiding (readFile, drop)
import Control.Monad
import Control.Monad.IO.Class
import Control.Exception
import Foreign.C.Types
import System.Random
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import SDL
import Debug.Trace
import qualified Linear
import Data.List.Split (chunksOf)

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
-- Initialised with random weights and biases as in genLayer.
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
window image width height = do
  initializeAll

  window <- createWindow "SDL Application" defaultWindow
  renderer <- createRenderer window (-1) defaultRenderer

  -- Convert to sdl texture
  texture <- loadTextureMNIST renderer width height image

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

-- Train a network using a given set of inputs and training values for the given
-- number of epochs using the given learning rate.
trainNetwork :: Network -> [Vector R] -> [Vector R] -> Int -> Int -> R -> Network
trainNetwork initialNet input trainingValues epochs batchSize learnRate =
  train initialNet 0
    where
      train net epoch = case epoch >= epochs of
        True  -> net
        False -> train nextNet (epoch + 1)
          where
            --inputBatch = fromRows input
            --trainingBatch = fromRows trainingValues
            inputBatches = map fromRows . chunksOf batchSize $ input
            trainingBatches = map fromRows . chunksOf batchSize $ trainingValues
            nextNet = foldl runBatch net (zip inputBatches trainingBatches)
            --nextNet = net
            runBatch network (inputBatch, trainingBatch) =
              gradientDescentM net inputBatch (map fst output) error learnRate
                where
                  output = evalNetworkM net inputBatch
                  error = evalNetworkErrorM net output inputBatch trainingBatch

toOutput = \n -> vector . map (\x -> if n == x then 1.0 else 0.0) $ [0..9]

fromOutput :: Vector R -> Int
fromOutput v = guess
  where
    list = zip [0..] . toList $ v
    possibles = filter ((>=0.5) . snd) list
    guess = case length possibles of
              0 -> -1
              1 -> fst . head $ possibles
              _ -> -2

runTest :: Network -> ([CUChar], Int) -> Bool
runTest network (image, label) = guess == label
  where output = fst . last $ evalNetwork network $ vector . map fromIntegral $ image
        guess = fromOutput output

--runTest :: Network -> ([CUChar], Int) -> IO ()
--runTest network (image, label) = do
--  let output = fst . last $ evalNetwork network $ vector . map fromIntegral $ image
--  let guess = fromOutput output
--  let expected = label
--  putStr "Guess: "
--  putStr . show $ guess
--  putStr ",\tExpected: "
--  putStr . show $ label
--  putStr $ if guess == expected then ".\tCorrect! " else ".\tINCORRECT!"
--  putStrLn ""

main = do

  (trainingImageCount, trainingWidth,
   trainingHeight, trainingImageData)  <- readMNISTImages "train-images-idx3-ubyte"
  (trainingLabelCount, trainingLabels) <- readMNISTLabels "train-labels-idx1-ubyte"

  (testImageCount, testWidth,
   testHeight, testImageData)  <- readMNISTImages "t10k-images-idx3-ubyte"
  (testLabelCount, testLabels) <- readMNISTLabels "t10k-labels-idx1-ubyte"

  assert (trainingWidth == testWidth && trainingHeight == testHeight) (return ())

  let input = take 60000 . map (vector . map fromIntegral) $ trainingImageData
  let trainingData = take 60000 . map toOutput $ trainingLabels
  
  initialNetwork <- genNetwork [20,10] (trainingWidth * trainingHeight)

  let trainedNetwork = trainNetwork initialNetwork input trainingData 2 10 3.0

  let testImage = head testImageData
  let testInput = vector . map fromIntegral $ testImage
  let testOutput = head testLabels --toOutput $ head testLabels

  --window testImage testWidth testHeight

  --let output = fst . last $ evalNetwork trainedNetwork testInput
  --let filteredOutput = map (>=0.5) . toList $ output

  --print $ fromOutput output
  --print $ testOutput

  let tests = map (runTest trainedNetwork) $ zip testImageData testLabels

  let success = length . filter (==True) $ tests
  let total = length tests

  putStr . show $ success
  putStr "/"
  putStrLn . show $ total

  --flip mapM_ (take 100 $ zip testImageData testLabels) runTest
  --  where
  --    runTest (imageData, label) = output
  --      where
  --        output2 = evalNetwork trainedNetwork
  --        output = putStrLn "test"
  --      --let output = evalNetwork trainedNetwork (vector . map fromIntegral $ imageData)
  --      --in do
  --      --  putStr "Guess: "
  --      --  putStr . show . fromOutput $ output
