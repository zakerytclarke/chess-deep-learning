# chess-deep-learning

Deep learning network trained off of stockfish games efficiently in memory

## Training Schedule



## Usage
```
from qchess import train_model, play_nn_stockfish, load_model
from sklearn.neural_network import MLPRegressor

model = train_model(
    model=MLPRegressor(hidden_layer_sizes = (800, 400, 800), random_state=1), 
    num_games=1000,
    depth=5,
    evaluate_every=100,
    mode="mixed",
    path=f"chess{depth}"
)


play_nn_stockfish(model,visualize=True)

```