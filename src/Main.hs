module Main where

import Control.Exception
import Control.Monad
import System.Random

-- A neural network input
type Input = Float

-- A neural network output
type Output = Float

-- A neural network weight
type Weight = Float

-- A single neuron with the given weights and bias
data Neuron = Neuron { weights :: [Float], bias :: Float } deriving Show

-- A layer of neurons
type Layer = [Neuron]

-- A neural network made up of layers of neurons
data Network = Network { layers :: [Layer] }

-- sigmoid
-- s-curve function, with value 0.5 at 0, tending to 1 at >0, and tending to 0 at <0.
sigmoid :: Float -> Float
sigmoid z = 1 / (1 + exp (-z))

-- evalNeuron
-- Evaluates a neuron against a set of inputs, returning the outputs.
evalNeuron :: Neuron -> [Input] -> Output
evalNeuron (Neuron weights bias) inputs = sigmoid (dot + bias)
  where
    dot = assert (length weights == length inputs) (sum . zipWith (*) weights $ inputs)

-- evalLayer
-- Takes a layer of neurons, and a list of inputs, and connects each input to
-- each neuron in the layer. It  then returns the result of applying each
-- neuron to each input.
evalLayer :: [Neuron] -> [Input] -> [Output]
evalLayer neurons inputs = map (flip evalNeuron inputs) neurons

-- evalNetwork
-- Takes a neural network and evalues it with the given input values.
evalNetwork :: Network -> [Input] -> [Output]
evalNetwork (Network layers) inputs = foldl (flip evalLayer) inputs layers

-- genNeuron
-- Generates a neuron with randomised weights and bias
genNeuron :: Int -> IO Neuron
genNeuron weights = do
  w <- replicateM weights (randomIO :: IO Weight)
  b <- randomIO :: IO Weight
  return $ Neuron w b

-- genLayer
-- Generates a layer of neurons with random biases
genLayer :: Int -> Int -> IO Layer
genLayer size weights = replicateM size (genNeuron weights)

main :: IO ()
main = do
  layer1 <- genLayer 10 10
  let network = Network [layer1]
  print $ evalNetwork network [0.1,0.2..1.0]
