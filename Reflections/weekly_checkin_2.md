# Weekly Check-in 2 by William Yue

* What work have you done?

This week, I spent almost all my time working on my attempt at a [Chess AI](https://github.com/willwin4sure/CSC_630_Machine_Learning/tree/main/ChessAI). This folder is pretty messy right now, but I'll clean it up soon.

I started by exploring the [python-chess](https://python-chess.readthedocs.io/en/latest/) library and documentation in order to figure out how to interact with it to play chess. I followed the format of my CSC 500 project on Connect Four, implementing a [chess_game.py](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/chess_game.py) file which ran the chess game between classes of players: [PlayerBot](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/player_bot.py) (which I should rename), [RandomBot](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/random_bot.py) (which plays random moves), and [TestingBot](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/testing_bot.py). 

TestingBot is the main one that I'm working on development with. I've managed to implement opening books, as well as a simple alpha-beta pruning algorithm. I'm still working on developing the evaluation function, which is the actual machine learning part, but have a simple material one coded up (sadly, I've also noticed that Python code is extremely slow in the tree search. No wonder everyone uses C++ for chess bots...). In [`model.py`](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/model.py), I have coded up a simple neural network which takes in FEN strings and their stockfish evaluations as a dataset, encodes the FEN strings, and feeds it through a neural network. The neural network performs ok, but I think I need to mess around with different board encodings and architecture for it to work better. 

* What videos of mine have you watched, if relevant? (*They're all posted on the homepage.*)

I've watched the Bias-Variance tradeoff video and the principle components analysis video.

* What external resources have you found the most useful? 

External documentation, random blog posts and videos on chess AI, and friends like *Ali Yang*.

* What questions do you have about the content? (*You should probably also share these with me in another way, because I might not always get to these check-ins in a timely manner. Still, it's helpful for you to reflect on them here.*)

Currently, I don't have any outstanding questions about the course content. If I did, I would come ask you during class or conference period, or send an email with a video!

* What are your goals for next week?

My goals for next week are to play around with the neural network in my Chess AI, as well as work through more of the PyTorch documentation. Hopefully I can get a decent bot working that can play decently quickly. Right now, the model still sucks a lot of the time (for example, white was up a bishop and rook and the model only thought it was +1).

* What are you submitting alongside this weekly check-in?

Nothing in the Canvas page, as all my work referenced is hosted on GitHub and linked in this check-in.

* Which other student(s) in class would you like to mention as being helpful toward your learning? How did they help?

I would like to thanik *Ali Yang* for helping me a ton with PyTorch and the neural network. 
