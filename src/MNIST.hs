module MNIST
  ( readMNISTImages
  , readMNISTLabels
  , loadTextureMNIST
  )
where

import Control.Exception
import qualified Data.ByteString.Lazy as BS
import Data.List.Split (chunksOf)
import Data.Binary.Get
import Foreign
import Foreign.C.Types
import SDL
import qualified Linear

-- Reads an MNIST data file into image data.
-- Returns the number of images, width, height, and a list of images, each of which is a list of CUChars.
readMNISTImages :: FilePath -> IO (Int, Int, Int, [[CUChar]])
readMNISTImages filename = do
  file <- BS.readFile filename

  let (magicNumber, numImages, numRows, numColumns) =
        flip runGet file $ do
          magicNumber <- getWord32be
          numImages <- getWord32be
          numRows <- getWord32be
          numColumns <- getWord32be
          return (magicNumber, numImages, numRows, numColumns)

  -- Ensure the magic number is 2051. It's some combination of the bytes: 0x00, 0x00, 0x08, 0x03
  -- Which are, in order, zero, zero, the data type (unsigned char), and the number of elements (3)
  assert (magicNumber == 2051) (return ())

  -- Read rest of file
  let rest = map CUChar . BS.unpack . BS.drop 16 $ file
  let images = chunksOf (fromIntegral (numRows * numColumns)) rest

  return (fromIntegral numImages, fromIntegral numRows, fromIntegral numColumns, images)

-- Load an nmist label file
readMNISTLabels :: FilePath -> IO (Int, [Int])
readMNISTLabels filename = do
  file <- BS.readFile filename

  let (magicNumber, numLabels) = flip runGet file $ do
        magicNumber <- getWord32be
        numLabels <- getWord32be
        return (magicNumber, numLabels)

  assert (magicNumber == 2049) (return ())

  -- Read labels
  let labels = BS.unpack . BS.drop 8 $ file

  return (fromIntegral numLabels, map fromIntegral labels)

-- Load an nmist image obtained from readMNISTImages into an SDL texture
loadTextureMNIST :: SDL.Renderer -> Int -> Int -> [CUChar] -> IO SDL.Texture
loadTextureMNIST renderer w h buf = do
  texture <- createTexture renderer RGBA8888 TextureAccessStreaming (Linear.V2 (fromIntegral w) (fromIntegral h))

  -- MNist image data is 1 byte, so duplicate each byte 4 times for 32-bit image format
  -- (RGB888 is 32 bit)
  let imageData = concatMap (replicate 4) buf

  (ptr, pitch) <- lockTexture texture Nothing
  let arr = castPtr ptr :: Ptr CUChar

  pokeArray arr imageData

  unlockTexture texture

  return texture
