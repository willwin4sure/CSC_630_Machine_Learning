# Weekly Check-in 1 by William Yue

* What work have you done?

This week, I continued working through the [first PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html) about the FashionMNIST dataset in this [Jupyter Notebook](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/Learn_PyTorch/PyTorch%20Tutorial.ipynb). While I have already picked up most of the material here through osmosis from documentation when working on my ChessAI, as well as through the help of peers like *Ali Yang*, it's great working through everything formally and making sure I understand exactly what's going on in every step.

I also continued to make progress on my [ChessAI](https://github.com/willwin4sure/CSC_630_Machine_Learning/tree/main/ChessAI). The bot still often makes quite terrible moves, but I"ve played around with a new bitboard piece-centric encoding as well as convolutional neural networks in this [file](https://github.com/willwin4sure/CSC_630_Machine_Learning/blob/main/ChessAI/testing_bot_conv.py) (note that the actual training code isn't in my repository yet, since I ran it on Google Colab to make use of the free GPU runtime that they provide). I'm going to work on trying out linear layers with this new encoding, and I"m also going to try playing around with the neural network architecture. I will also try the "Pos2Vec" approach of a certain paper I saw online, if I can figure out how to get Deep Belief neural networks to work, as well as playing around with seeing which positions my model fails to classify well by plotting out the errors or some confusion matrix.

* What videos of mine have you watched, if relevant? (*They're all posted on the homepage.*)

At this point, I've watched all the videos posted on the Canvas page. This week, I finished the "Complete" Data Modeling Process video.

* What external resources have you found the most useful? 

Again, mostly just online documentation like the PyTorch tutorials or StackOverflow/blog posts if I need help. I've also been consulting papers regarding Chess AI for inspiration and some GitHub repositories for how other people have implemented them.

* What questions do you have about the content? (*You should probably also share these with me in another way, because I might not always get to these check-ins in a timely manner. Still, it's helpful for you to reflect on them here.*)

Currently, I don't have any outstanding questions about the course content. If I did, I would come ask you during class or conference period, or send an email with a video!

* What are your goals for next week?

My goals for next week are to finish up my Chess AI with Davin (doing all the things I've mentioned above about playing with the approach, model architecture, dataset, encodings, etc. as well as analyzing the computer's mistakes) and hopefully get it in a better working condition, as well as finishing up the PyTorch introduction tutorial. I will also finally finish up my black box reflection research and turn in the Gradients project, hopefully early in the week (somethign I've been procrastinating on for a while).

* What are you submitting alongside this weekly check-in?

Nothing in the Canvas page, as all my work referenced is hosted on GitHub and linked in this check-in.

* Which other student(s) in class would you like to mention as being helpful toward your learning? How did they help?

*Davin Jeong* has been working with me on the tree search part of the AI, which has been fantastic. 