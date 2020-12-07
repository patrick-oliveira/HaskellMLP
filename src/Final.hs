{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Final where 

import qualified Data.Matrix as M
import Data.Random ()
import Data.Random.Distribution (Distribution)
import Data.Random.Distribution.Uniform (stdUniform, StdUniform)
import Data.Random.Distribution.Normal (stdNormal, Normal)
import Control.Monad.State (forM, replicateM, evalState)
import Data.RVar (sampleRVar)
import System.Random (randomR, StdGen, mkStdGen)
import Numeric.AD (auto, grad)
import Data.Matrix (multStd)
import Text.Printf (printf)
import Control.Monad.ST (runST, ST)
import GHC.Arr (STArray)
import Data.STRef (writeSTRef, readSTRef, newSTRef)
import Data.Array.ST ( STArray, newListArray, readArray, writeArray ) 

{- General Types -}
-- Facilitar a leitura das funções que usam pesos e gradientes
type Weights a = M.Matrix a
type Grads a   = M.Matrix a

-- Facilitar a leitura das funções que envolvem input/output
type Input a   = M.Matrix a
type Output a  = M.Matrix a
type Target a  = M.Matrix a
type Sample a = [(Input a, Target a)]

type InputZ a  = M.Matrix a
type InputY a  = M.Matrix a
type OutputZ a = M.Matrix a
type OutputY a = M.Matrix a





{- Funções Auxiliares -}
-- Amostragem da var. uniforme
nRandomUniform :: (Distribution StdUniform a, Floating a) => Int -> [a]
nRandomUniform n = evalState (replicateM n (sampleRVar stdUniform)) (mkStdGen seed)
    where
        seed = 0

-- Amostragem da var. normal
nRandomNormal :: (Distribution Normal a, Floating a) => Int -> [a]
nRandomNormal n = evalState (replicateM n (sampleRVar stdNormal)) (mkStdGen seed)
    where
        seed = 1

-- Amostragem de matrizes aleatorias
randomMatrix :: (Distribution Normal a, Floating a) => Int -> Int -> M.Matrix a
randomMatrix m n = M.fromList m n (nRandomNormal (m*n))

shuffle' :: [a] -> StdGen -> ([a],StdGen)
shuffle' xs gen = runST (do
        g <- newSTRef gen
        let randomRST lohi = do
              (a,s') <- fmap (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1..n] $ \i -> do
                j <- randomRST (i,n)
                vi <- readArray ar i
                vj <- readArray ar j
                writeArray ar j vi
                return vj
        gen' <- readSTRef g
        return (xs',gen'))
  where
    n = length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs =  newListArray (1,n) xs












{-  Activation Functions    -}
-- Funções de ativaćão
data ActivationFunction = Sigmoid | ReLU | Tanh | SigmoidClassification | Id
    deriving (Show)
activate :: (Ord a, Num a, Floating a) => ActivationFunction -> a -> a 
activate Id x = x
activate Sigmoid x = 1.0 / (1.0 + exp(-x))
activate Tanh x = (exp(2*x) - 1) / (exp(2*x) + 1)
activate ReLU x = max 0.0 x


{-
    TODO: - Criar uma função para gerar one-hot encodings a partir de um vetor/distribuićão de probabilidades
-}
-- softMax :: (Floating a) => Input a -> Output a
-- softMax x = fmap (/expSum) expVector
--     where
--         expVector = fmap exp x
--         expSum = sum expVector














{- Error Functions -}
data Loss = SE | BCE
getLoss :: Floating a => Loss -> (M.Matrix a -> M.Matrix a -> a)
getLoss SE = sqrdError
getLoss BCE = binaryCrossEntropy

sqrdError :: Floating a => M.Matrix a -> M.Matrix a -> a
sqrdError y ŷ = sum (fmap (^2) (ŷ - y)) / l
    where
        l = fromIntegral $ length y

binaryCrossEntropy :: Floating a => M.Matrix a -> M.Matrix a -> a
binaryCrossEntropy y ŷ = - sum (zipWith f y' ŷ')
    where
        f = \x y -> x * log y + (1 - x) * log (1 - y)
        y' = M.toList y
        ŷ' = M.toList ŷ

calcLoss :: (Ord a, Floating a) => (Input a, Target a) -> Loss -> Network a ->  a
calcLoss (x, y) loss network = e y (estimate network x)
    where
        e = getLoss loss

calcMeanLoss :: (Ord a, Floating a) => [Input a] -> [Target a] -> Loss -> Network a -> a
calcMeanLoss xs ys loss network =  errorSum/l
    where
        samples = zip xs ys
        samplesErrors = map (\s -> calcLoss s loss network) samples
        errorSum = sum samplesErrors
        l = fromIntegral (length samplesErrors)
















{-  Neural Networks Structures  -}
-- Rede genérica
newtype Network a = Network [Component a]
                    deriving (Functor, Foldable, Traversable)
{-
    A rede precisa ser uma instância de Traversable para que seja uma variável válida para a aplicação do autograd.
    TODO: - Descobrir como instanciar manualmente (i.e. como funcionam as funções) o Foldable e o Traversable para o tipo Network.
-}

-- instance Functor Network where
    -- fmap f (Netwok layers) = Network (map (fmap f) layers)

instance (Show a) => Show (Network a) where
    show (Network layers) = show layers

instance Num a => Num (Network a) where 
    (Network layers1) - (Network layers2) = Network (zipWith (-) layers1 layers2)
    (Network layers1) + (Network layers2) = Network (zipWith (+) layers1 layers2) 
    _ * _ = undefined 
    abs _ = undefined
    signum _ = undefined
    fromInteger _ = undefined
{-
    A instância de Num é definida para facilitar a correção da rede com o gradiente.
    As definições presumem que as redes possuem estruturas idênticas, o que pode incorrer em erro.
    Posso trocar essas soma/subtração por funções seguras ou definir mecanismos de garantia em outro lugar, como na criação das redes.
    É garantido que só há uma soma/subtração na correção com o gradiente, então as estruturas serão, de fato, sempre iguais. 
-}
forward :: (Ord a, Floating a) => Network a -> Input a -> [(OutputZ a, OutputY a)]
forward (Network layers) x = scanl (\x c -> forwardApply (triggerComponent c) x) (x, x) layers

estimate :: (Ord a, Floating a) => Network a -> Input a -> Output a
estimate net x = snd (last (forward net x))

estimateAll :: (Ord a, Floating a) => Network a -> [Input a] -> [Output a]
estimateAll net = map (estimate net)

-- classify :: Floating a => Network a -> Input a -> Output a
-- classify net x = oneHotEncode probDistribution
--     where
--         probDistribution = softMax (estimate net x)
--         oneHotEncode :: Floating a => Input a -> Input a
--         oneHotEncode x = undefined
--             where
--                 l = length x

costGrad :: (Ord a, Floating a) => Network a -> (Input a, Target a) -> Loss -> Network a
costGrad network (input, target) loss = grad (calcLoss (fmap auto input, fmap auto target) loss) network

stepLearning :: (Ord a, Floating a) => Network a -> (Input a, Target a) -> Loss -> a -> Network a
stepLearning network (input, target) loss lrate = network - fmap (*lrate) gradient
    where
        gradient = costGrad network (input, target) loss

stochasticLearning :: (Ord a, Floating a) => (Network a, Sample a) -> Loss -> a -> (Network a, Sample a)
stochasticLearning (network, samples) loss lrate = (foldl (\net s -> stepLearning net s loss lrate) network samples', samples')
    where
        samples' = fst $ shuffle' samples (mkStdGen seed)
        seed = 0







-- Componente genérica da rede
data Component a =  Component { layers :: [Weights a]
                              , activationFunction :: ActivationFunction
                              , propagate :: PropagationFunction} 
                              deriving (Functor, Foldable, Traversable)

instance Num a => Num (Component a) where -- Presumindo que as componentes poessuem estruturas idênticas
    component1 - component2 = Component { layers = zipWith (-) (layers component1) (layers component2)
                                        , activationFunction = activationFunction component1
                                        , propagate = propagate component1}
            
    component1 + component2 = Component { layers = zipWith (+) (layers component1) (layers component2)
                                        , activationFunction = activationFunction component1
                                        , propagate = propagate component1}
    _ * _ = undefined 
    abs _ = undefined
    signum _ = undefined
    fromInteger _ = undefined

instance Show a => Show (Component a) where
    show c = "\n"++ "Componente: " ++ show t ++ "\n"
                 ++ "Ativação: "   ++ show a ++ "\n"
                 ++ "Pesos = \n"       ++ printWeights w
        where
            t = propagate c
            a = activationFunction c
            w = layers c

            printWeights [] = ""
            printWeights (w:ws) = show w ++ "\n" ++ printWeights ws 


triggerComponent :: (Ord a, Floating a) => Component a -> ((InputZ a, InputY a) -> (OutputZ a, OutputY a))
triggerComponent c = triggeredComponent
    where
        f = propagate c
        w = layers c
        a = activationFunction c 
        triggeredComponent = getPropagationFunction f a w -- (M.Matrix a, M.Matrix a) -> (M.Matrix a, M.Matrix a)

forwardApply :: ((InputZ a, InputY a) -> (OutputZ a, OutputY a)) -> (InputZ a, InputY a) -> (OutputZ a, OutputY a)
forwardApply f = f

data PropagationFunction = Linear | Pass
    deriving (Show)
    
getPropagationFunction :: (Ord a, Floating a) => PropagationFunction -> (ActivationFunction -> [Weights a] -> ((InputZ a, InputY a) -> (OutputZ a, OutputY a)))
-- z : combinação linear
-- y : ativação
getPropagationFunction Pass = \_ _ x -> x
getPropagationFunction Linear = \a [w, b] i -> let (x, z) = (snd i, multStd w x + b) 
                                                in (z, fmap (activate a) z)


-- Detalhes das layers
_Linear :: (Floating a) => (Weights a, Weights a) -> ActivationFunction -> Component a
_Linear (w, b) a = Component { layers = [w, b]
                             , activationFunction = a
                             , propagate = Linear}











{--
    Quanto aos inputs, o programa presume que os dados estão organizados em dois arquivos, um para features e um para targets, onde colunas são as variáveis e linhas são as amostras, cada informação separada por " ".
--}
dataToMatrix :: String -> [M.Matrix Double]
dataToMatrix file = map (M.fromList numFeatures 1) matrixDoubles 
    where
        matrixDoubles = map (map (\x -> read x :: Double) . words) (lines file) 
        numFeatures = length (head matrixDoubles)