# Weekly Check-in 1 by William Yue

* What work have you done?

This week, I've worked on several projects.

In terms of the [Gradients Project](https://github.com/willwin4sure/CSC_630_Machine_Learning/tree/main/Gradients_Project), I noticed that there were several errors that popped up when running logistic regression, associated with domain issues in taking logarithms of nonpositive numbers and division by zero. I believe that these are likely due to the fact that some values get incredibly close to zero and Python may round them down to zero. To solve this issue, I implemented several methods in the `Variable` class in [`variable.py`](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/Gradients_Project/variable.py), specifically `safe_eval()` and `nonzero()`, which help prevent these errors (I may think a bit more about how to implement them properly, like retaining the correct sign for `nonzero()`). I also changed the loss `threshold` of the `fit` function of my `LogisticRegression` class in [`logistic_regression.py`](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/Gradients_Project/logistic_regression.py) to a `total_times` hyperparameter, which would specify precise the total number of times to iterate gradient descent. I also stored the best loss and input parameters throughout the whole process. I hope to continue working on some other methods to avoid falling into local minima (like looking at different starting points).

I've also started learning PyTorch, in this [Jupyter Notebook](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/Learn_PyTorch/PyTorch%20Tutorial.ipynb), and I plan for this to be my main project in the next couple of weeks. I may also explore building a rudimentary chess robot (exploring things like decision trees, and looking into understanding how things like [Leela Chess Zero](https://lczero.org/) work). So far, all I've done is [explore the `python-chess` library](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/chess_library.ipynb) a bit which will have the rules of the game and methods of displaying the board hard coded already.

I also spend some time setting up CUDA and running models on the NVIDIA GPUs in the Makerspace computers.

* What videos of mine have you watched, if relevant? (*They're all posted on the homepage.*)

I haven't looked at the videos posted on the front page yet, but I plan on doing so in this coming weekend.

* What external resources have you found the most useful? 

Mostly external documentation, like [this one](https://python-chess.readthedocs.io/en/latest/) about the `python-chess` library, as well as several [tutorials](https://pytorch.org/tutorials/) on the PyTorch website. I also search up tutorials online for installing things (like figuring out how CUDA and the NVIDIA GPUs work on the DataLab computers), and commonly solve problems I run into using posts on StackOverflow, etc.

* What questions do you have about the content? (*You should probably also share these with me in another way, because I might not always get to these check-ins in a timely manner. Still, it's helpful for you to reflect on them here.*)

Currently, I don't have any outstanding questions about the course content. If I did, I would come ask you during class or conference period, or send an email with a video!

* What are your goals for next week?

My goals for next week are to first finish the Gradients Project (hopefully over the weekend), explore some of Dr. Z's videos posted on the Canvas page, and then continue learning PyTorch in my Jupyter notebook. I will also write up something quick about how to install a lot of the tools used in machine learning, as well as maybe some quick tutorials on how everything works (so it's understandable even to a beginner).

* What are you submitting alongside this weekly check-in?

Nothing in the Canvas page, as all my work referenced is hosted on GitHub and linked in this check-in.

* Which other student(s) in class would you like to mention as being helpful toward your learning? How did they help?

I would like to thank *Arnav Bhakta* for his continual support in working with me on the LSTM project. I would also like to thank *Ali Yang* for talking about PyTorch with me and *Carissa Yip* for talking about chess with me.

