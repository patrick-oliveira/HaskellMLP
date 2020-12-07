module LinearSeparation where

import Final
import Text.Printf (printf)

linear :: IO()
linear = do
    x_raw <- readFile "src/datasets/artificial_linear/x.txt"
    y_raw <- readFile "src/datasets/artificial_linear/y.txt"
    
    let x = dataToMatrix x_raw
        y = dataToMatrix y_raw
        samples = zip x y

    let trainLoops = 100
        learningRate = 0.1
        lossFunction = BCE

    let l1 = _Linear (randomMatrix 1 2, randomMatrix 1 1) Sigmoid
        net = Network [l1]
        netUpdated = fst $ iterate (\xs -> stochasticLearning xs lossFunction learningRate) (net, samples) !! trainLoops 

    let loss1, loss2 :: Double
        loss1 = calcMeanLoss x y lossFunction net
        loss2 = calcMeanLoss x y lossFunction netUpdated
    printf "\nErro antes do Treinamento : %5.5f" loss1
    printf "\nErro após o treinamento : %5.5f" loss2

    print ""

    print "Algumas predicoes da rede inicial:"
    print (map (fmap round) (estimateAll net (take 10 x)))
    print "Predicoes correspondentes da rede final:"
    print (map (fmap round) (estimateAll netUpdated (take 10 x)))
    print "Valores esperados :"
    print (take 10 y)

    return ()