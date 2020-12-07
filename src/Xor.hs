module Xor where

import qualified Data.Matrix as M
import Final
import Text.Printf (printf)

xor :: IO()
xor = do
    let x = map (M.fromList 2 1) [[1, 0], [1, 1], [0, 1], [0, 0]]
        y = map (M.fromList 1 1) [[1],    [0],    [1],    [0]]
        samples = zip x y

    let trainLoops = 1000
        learningRate = 0.5
        lossFunction = BCE

    let l1 = _Linear (randomMatrix 2 2, randomMatrix 2 1) Sigmoid
        l2 = _Linear (randomMatrix 1 2, randomMatrix 1 1) Sigmoid
        net = Network [l1, l2]
        netUpdated = fst $ iterate (\x -> stochasticLearning x lossFunction learningRate) (net, samples) !! trainLoops

    let loss1, loss2 :: Double
        loss1 = calcMeanLoss x y lossFunction net
        loss2 = calcMeanLoss x y lossFunction netUpdated

    printf "\nErro antes do Treinamento: %5.5f" loss1
    printf "\nErro após o treinamento : %5.5f" loss2

    print ""

    print "Predicoes da rede inicial:"
    print (map (fmap round) (estimateAll net x))
    print "Predicoes da rede final:"
    print (map (fmap round) (estimateAll netUpdated x))

    return ()