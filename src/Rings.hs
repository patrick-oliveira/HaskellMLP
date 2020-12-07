module Rings where 

import Final
import Text.Printf (printf)

rings :: IO()
rings = do
    x_raw <- readFile "src/datasets/artificial_rings/x.txt"
    y_raw <- readFile "src/datasets/artificial_rings/y.txt"
    
    let x = dataToMatrix x_raw
        y = dataToMatrix y_raw
        samples = zip x y

    let trainLoops = 100
        learningRate = 0.05
        lossFunction = BCE

    let l1 = _Linear (randomMatrix 10 2, randomMatrix 10 1) ReLU
        l2 = _Linear (randomMatrix 10 10, randomMatrix 10 1) ReLU
        l3 = _Linear (randomMatrix 1 10, randomMatrix 1 1) Sigmoid
        net = Network [l1, l2, l3]
        netUpdated = fst $ iterate (\xs -> stochasticLearning xs lossFunction learningRate) (net, samples) !! trainLoops
 
    let loss1, loss2 :: Double
        loss1 = calcMeanLoss x y lossFunction net
        loss2 = calcMeanLoss x y lossFunction netUpdated

    printf "\nErro antes do Treinamento: %5.3f" loss1
    printf "\nErro após o treinamento : %5.3f" loss2

    print "\nPredicoes da rede inicial:"
    print (map (fmap round) (estimateAll net (take 10 x)))
    print "\nPredicoes da rede final:"
    print (map (fmap round) (estimateAll netUpdated (take 10 x)))
    print "\nValores esperados:"
    print (take 10 y)       

    return ()