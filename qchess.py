import chess
import chess.engine
import chess.svg
import random
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PATH = '/dbfs/FileStore/zak/'

METRICS = pd.DataFrame({
    'iteration':[],
    'rmses':[],
    'cdes':[]
})


def material_balance(board):
    if board.is_game_over():
        if board.is_checkmate():
            if board.turn: # Backwards since the player in check has lost
                return -100
            else:
                return 100
        else: # Stalemate
            return 0

    # https://github.com/niklasf/python-chess/discussions/864
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return (
        chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
        3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
        3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
        5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
        9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
    )



def board_representation(board):
    """Convert a chess board to nn representation
    64x12- does piece x exists at position y
    1- turn
    1- depth (0-20)
    """
    out = []
    for i in range(0,64): # For every position
        for p in ["p","r","n","b","q","k","P","R","N","B","Q","K"]: # For every piece
            piece = board.piece_at(i)
            if piece:
                out.append(1 if piece.symbol()==p else 0)
            else:
                out.append(0)
    out.append(1 if board.turn else 0)
    return out

def move_representation(move):
    from_move = [0]*64
    from_move[move.from_square] = 1
    to_move = [0]*64
    to_move[move.to_square] = 1
    return from_move + to_move

def shift_scores(scores, n):
    return scores[n:]+([scores[-1]]*n)[0:-1]

def generate_game(engine, moves="random"):
    encodings = []
    scores = []
    board = chess.Board()
    while not board.is_game_over():
        board_rep = board_representation(board)
        scores.append(material_balance(board))

        if moves=="engine":
            result = engine.play(board, chess.engine.Limit(depth=random.randint(1,20)))
            encodings.append(board_rep + move_representation(result.move))
            board.push(result.move)
        elif moves=="random":
            random_move = random.choice(list(board.generate_legal_moves()))
            encodings.append(board_rep + move_representation(random_move))
            board.push(random_move)
        elif moves=="mixed":
            if random.randint(0,2)>1:
                result = engine.play(board, chess.engine.Limit(depth=random.randint(1,20)))
                move = result.move
            else:
                move = random.choice(list(board.generate_legal_moves()))
            encodings.append(board_rep + move_representation(move))
            board.push(move)

    # Score
    scores.append(material_balance(board))

    return encodings, scores


def generate_dataset(num_games, depth_offset):
    encodings = []
    scores = []
    
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")

    for i in range(0,num_games):
        es, ss = generate_game(engine)
        encodings.extend(es)
        scores.extend(shift_scores(ss, depth_offset))

    engine.quit()

    return encodings, scores


def load_model(path):
    return pickle.load(open(path, 'rb'))

def get_model_move(model, board):
    best_move = None
    best_score = -100
    
    possible_moves = list(board.generate_legal_moves())
    predicted_moves = sorted(zip(possible_moves, list(map(lambda x:model.predict([board_representation(board) + move_representation(x)])[0],possible_moves))),key=lambda tup: tup[1])
    best_move = predicted_moves[-1][0]
    best_score = predicted_moves[-1][1]
    print(best_move, best_score, material_balance(board))
    return best_move, predicted_moves


def train_model(model=MLPRegressor(hidden_layer_sizes = (800, 400, 800), random_state=1), num_games=100, depth_offset=1, evaluate_every=10):
    """Simultaneously create and train neural network to more 
    advantageously use memory consumption
    """
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
    
    print("Chess Q-Learning")
    print("=========================")
    print("Creating Evaluation Dataset")
    X_test, y_test =  generate_dataset(10, depth_offset)
    
    global METRICS

    print("Begin Training")
    for i in range(0, num_games):
        try:
            print(i)
            es, ss = generate_game(engine)
            encodings = es
            scores = shift_scores(ss, depth_offset)
            model = model.partial_fit(encodings, scores)

            if i % evaluate_every == 0:
                print("=========================")
                pickle.dump(model, open(PATH+'chess.pkl', 'wb'))
                game_material = play_nn_stockfish(model)
                print(f"Update: {i} game iterations")
                cde = model.score(X_test, y_test)
                rmse = math.sqrt(mean_squared_error(y_test,model.predict(X_test)))
                print(f"r^2: {cde}")
                print(f"RMSE: {rmse}")
                print(f"Sample Game Material Score:",game_material)
                # Show Graph of Training errors
                METRICS = METRICS.append(pd.DataFrame({
                    'iteration':[i],
                    'rmses':[rmse],
                    'cdes':[cde]
                }), ignore_index = True)
                METRICS.to_csv(PATH+"metrics.csv")
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add traces
                fig.add_trace(
                    go.Scatter(x=METRICS['iteration'], y=METRICS['cdes'], name="Coef Det"),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(x=METRICS['iteration'], y=METRICS['rmses'], name="RMSE"),
                    secondary_y=True,
                )

                # Add figure title
                fig.update_layout(
                    title_text="Chess Material Prediction Depth=1"
                )

                # Set x-axis title
                fig.update_xaxes(title_text="Game Iterations")

                # Set y-axes titles
                
                fig.show()


        
        except:
            print()
        
    engine.quit()
    
    return regr




def play_nn_stockfish(model, depth=5, visualize=False):
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
    board = chess.Board()
    count = 0
    while not board.is_game_over():        
        nn_move, all_moves = get_model_move(model, board)
        
        print(board)
        print("")
        
        if visualize:
            f = open(f"./output/view{count}.svg","w")
            f.write(chess.svg.board(board, arrows=list(map(lambda x:chess.svg.Arrow(x[0].from_square, x[0].to_square, color="#0000cc"+["66","77","88","99","aa","bb","cc","dd","ee","ff"][round(x[1]/100*10)]),filter(lambda x:x[1]>0,all_moves)))+list(map(lambda x:chess.svg.Arrow(x[0].from_square, x[0].to_square, color="#cc0000"+["66","77","88","99","aa","bb","cc","dd","ee","ff"][round(x[1]/100*10)]),filter(lambda x:x[1]<0,all_moves))), size=350) )
            f.close()
            count = count + 1


        board.push(nn_move)

        stockfish_result = engine.play(board, chess.engine.Limit(depth=depth))
        board.push(stockfish_result.move)
    
    engine.quit()
    return material_balance(board)

