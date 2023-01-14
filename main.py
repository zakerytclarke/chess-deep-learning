import chess
import chess.engine
import chess.svg
import random
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pickle
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TODO: modify to include check and end game states
def material_balance(board):
    if board.is_game_over():
        if board.is_checkmate():
            if board.turn:
                return 100
            else:
                return -100
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

def generate_game(engine, moves="engine"):
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
        # print(board)
        # print("")

    # Score
    scores.append(material_balance(board))

    return encodings, scores


def generate_dataset(num_games, depth_offset):
    encodings = []
    scores = []
    
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")

    for i in range(0,num_games):
        print(i)
        es, ss = generate_game(engine)
        encodings.extend(es)
        scores.extend(shift_scores(ss, depth_offset))

    engine.quit()

    return encodings, scores



def create_dataset():

    dataset_x, dataset_y = generate_dataset(num_games = 1000, depth_offset = 5)

    f = open("dataset_x.json", "w")
    f.write(json.dumps(dataset_x))
    f.close()

    f = open("dataset_y.json", "w")
    f.write(json.dumps(dataset_y))
    f.close()


def train_model(dataset_x, dataset_y):    
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes = (400, 100), random_state=1).fit(X_train, y_train)

    # regr.predict(X_test[:2])
    pickle.dump(model, open('model.pkl', 'wb'))
    print(regr.score(X_test, y_test))

    return regr

def load_model(path):
    return pickle.load(open(path, 'rb'))

def get_model_move(board):
    best_move = None
    best_score = -100
    
    possible_moves = list(board.generate_legal_moves())
    predicted_moves = sorted(zip(possible_moves, list(map(lambda x:model.predict([board_representation(board) + move_representation(x)])[0],possible_moves))),key=lambda tup: tup[1])
    best_move = predicted_moves[-1][0]
    best_score = predicted_moves[-1][1]
    print(best_move, best_score, material_balance(board))
    return best_move, predicted_moves

def create_train_dataset(num_games=100, depth_offset=10):
    """Simultaneously create and train neural network to more 
    advantageously use memory consumption
    """
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
    regr = MLPRegressor(hidden_layer_sizes = (400, 100), random_state=1)

    # regr.predict(X_test[:2])
    
    for i in range(0,int(num_games*0.7)):
        print(i)
        es, ss = generate_game(engine)
        encodings = es
        scores = shift_scores(ss, depth_offset)
        regr = regr.partial_fit(encodings, scores)

        if i % 100 == 0:
            pickle.dump(regr, open('model.pkl', 'wb'))

    
    X_test, y_test =  generate_dataset(int(num_games*0.3),10)
    print(regr.score(X_test, y_test))
    
    pickle.dump(regr, open('model.pkl', 'wb'))

    engine.quit()

    

    return regr




def play_nn_stockfish():
    
    engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")

    board = chess.Board()
    
    count = 0
    while not board.is_game_over():        
        nn_move, all_moves = get_model_move(board)
        
        print(board)
        print("")

        f = open(f"./output/view{count}.svg","w")
        f.write(chess.svg.board(board, arrows=list(map(lambda x:chess.svg.Arrow(x[0].from_square, x[0].to_square, color="#0000cc"+["66","77","88","99","aa","bb","cc","dd","ee","ff"][round(x[1]/100*10)]),filter(lambda x:x[1]>0,all_moves)))+list(map(lambda x:chess.svg.Arrow(x[0].from_square, x[0].to_square, color="#cc0000"+["66","77","88","99","aa","bb","cc","dd","ee","ff"][round(x[1]/100*10)]),filter(lambda x:x[1]<0,all_moves))), size=350) )
        f.close()
        count = count + 1


        board.push(nn_move)

        stockfish_result = engine.play(board, chess.engine.Limit(depth=10))
        board.push(stockfish_result.move)
        
    engine.quit()

# create_dataset()

# f = open("dataset_x_1000.json")
# dataset_x = json.load(f)
# f.close()
# f = open("dataset_y_1000.json")
# dataset_y = json.load(f)
# f.close()

# # model = train_model(dataset_x,dataset_y)
# model = load_model("model_1000.pkl")
#print(math.sqrt(mean_squared_error(dataset_y,model.predict(dataset_x))))


model = create_train_dataset()

play_nn_stockfish()

