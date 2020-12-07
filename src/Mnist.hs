module Mnist where

import Final
import Text.Printf (printf)

mnist :: IO()
mnist = do
    x_raw <- readFile "src/datasets/mnist/x.txt"
    
    let x = dataToMatrix x_raw
        samples = take 50 (zip x x)

    let trainLoops = 1
        learningRate = 0.1
        lossFunction = SE

    let l1 = _Linear (randomMatrix 300 784, randomMatrix 300 1) ReLU
        l2 = _Linear (randomMatrix 784 300, randomMatrix 784 1) ReLU
        net = Network [l1, l2]
        netUpdated = fst $ iterate (\xs -> stochasticLearning xs lossFunction learningRate) (net, samples) !! trainLoops 

    let loss1, loss2 :: Double
        loss1 = calcMeanLoss x x lossFunction net
        loss2 = calcMeanLoss x x lossFunction netUpdated
    printf "\nErro antes do Treinamento : %5.3f" loss1
    printf "\nErro após o treinamento : %5.3f \n" loss2

    return ()