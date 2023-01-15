from qchess import train_model, play_nn_stockfish
from sklearn.neural_network import MLPRegressor


    

model = train_model(
    model=MLPRegressor(hidden_layer_sizes = (800, 400, 800), random_state=1), 
    num_games=10000,
    depth_offset=1,
    evaluate_every=1
)

play_nn_stockfish(model)
