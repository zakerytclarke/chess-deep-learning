from qchess import train_model, play_nn_stockfish, load_model
import qchess
from sklearn.neural_network import MLPRegressor


model = MLPRegressor(hidden_layer_sizes = (800, 400, 800), random_state=1)
# training_schedule = {
#     'depth':[1,2,3,4,5],
#     'num_games':[10000,5000,2000,1000,1000],
#     'mode':["random","random","mixed","mixed","enginerandom"],
# }
training_schedule = {
    'depth':[1,2,3,4,5],
    'num_games':[1,1,1,1,1],
    'mode':["random","random","mixed","mixed","enginerandom"],
}
for i in range(0,len(training_schedule.get('depth'))):
    depth = training_schedule.get('depth')[i]
    model = train_model(
        model=model, 
        num_games=training_schedule.get('num_games')[i],
        depth=depth,
        evaluate_every=1,
        mode=training_schedule.get('mode')[i],
        path=f"chess{depth}"
    )


play_nn_stockfish(model,visualize=True)
