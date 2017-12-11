

<div style="text-align:center"><img src ="./images/local/header.png" /></div>

This repository contains the code and pdf of a series of blog post called "dissecting reinforcement learning" which I published on my blog [mpatacchiola.io/blog](https://mpatacchiola.github.io/blog/). Moreover there are links to resources that can be useful for a reinforcement learning practitioner. **If you have some good references which may be of interest please send me a pull request and I will integrate them in the README**.

The source code is contained in [src](./src) with the name of the subfolders following the post number. In [pdf](./pdf) there are the A3 documents of each post for offline reading. In [images](./images) there are the raw svg file containing the images used in each post.

Installation
------------

The source code does not require any particular installation procedure. The code can be used in **Linux**, **Windows**, **OS X**, and embedded devices like **Raspberry Pi**, **BeagleBone**, and **Intel Edison**. The only requirement is *Numpy* which is already present in Linux and can be easily installed in Windows and OS X through [Anaconda](https://conda.io/docs/install/full.html) or [Miniconda](https://conda.io/miniconda.html). Some examples require [Matplotlib](https://matplotlib.org/) for data visualization and animations.

Posts Content
------------

1. [[Post one]](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) [[code]](./src/1) [[pdf]](./pdf) - Markov chains. Markov Decision Process. Bellman Equation. Value and Policy iteration algorithms. 

2. [[Post two]](https://mpatacchiola.github.io/blog/2017/01/15/dissecting-reinforcement-learning-2.html) [[code]](./src/2) [[pdf]](./pdf) - Monte Carlo methods for prediction and control. Generalised Policy Iteration. Action Values and Q-function.

3. [[Post three]](https://mpatacchiola.github.io/blog/2017/01/29/dissecting-reinforcement-learning-3.html) [[code]](./src/3) [[pdf]](./pdf) - Temporal Differencing Learning, Animal Learning, TD(0), TD(λ) and Eligibility Traces, SARSA, Q-Learning.

4. [[Post four]](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html) [[code]](./src/4) [[pdf]](./pdf) - Neurobiology behind Actor-Critic methods, computational Actor-Critic methods, Actor-only and Critic-only methods.

5. [[Post five]](https://mpatacchiola.github.io/blog/2017/03/14/dissecting-reinforcement-learning-5.html) [[code]](./src/5) [[pdf]](./pdf) - Evolutionary Algorithms introduction, Genetic Algorithm in Reinforcement Learning, Genetic Algorithms for policy selection.

6. [[Post six]](https://mpatacchiola.github.io/blog/2017/08/14/dissecting-reinforcement-learning-6.html) [[code]](./src/6) [[pdf]](./pdf) - Reinforcement learning applications, Multi-Armed Bandit, Mountain Car, Inverted Pendulum, Drone landing, Hard problems.

6. [[Post seven]](https://mpatacchiola.github.io/blog/2017/12/11/dissecting-reinforcement-learning-7.html) [[code]](./src/7) [[pdf]](./pdf) - Function approximation, Intuition, Linear approximator, Applications, High-order approximators.


Resources
---------

**Software:**

- [[Google DeepMind Lab]](https://deepmind.com/blog/open-sourcing-deepmind-lab/) [[github]](https://github.com/deepmind/lab) - DeepMind Lab is a fully 3D game-like platform tailored for agent-based AI research.

- [[OpenAI Gym]](https://gym.openai.com/) [[github]](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms.

- [[OpenAI Universe]](https://universe.openai.com/) [[github]](https://github.com/openai/universe) - Measurement and training for artificial intelligence.

- [[RL toolkit]](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/RLtoolkit1.0.html) - Collection of utilities and demos developed by the RLAI group which may be useful for anyone trying to learn, teach or use reinforcement learning (by Richard Sutton).

- [[setosa blog]](http://setosa.io/blog/2014/07/26/markov-chains/index.html) - A useful visual explanation of Markov chains.  

**Books and Articles:**

- **Artificial intelligence: a modern approach. (chapters 17 and 21)** Russell, S. J., Norvig, P., Canny, J. F., Malik, J. M., & Edwards, D. D. (2003). Upper Saddle River: Prentice hall. [[web]](http://aima.cs.berkeley.edu/) [[github]](https://github.com/aimacode)

- **Christopher Watkins** doctoral dissertation, which introduced the **Q-learning** for the first time [[pdf]](https://www.researchgate.net/profile/Christopher_Watkins2/publication/33784417_Learning_From_Delayed_Rewards/links/53fe12e10cf21edafd142e03/Learning-From-Delayed-Rewards.pdf)

- **Evolutionary Algorithms for Reinforcement Learning.** Moriarty, D. E., Schultz, A. C., & Grefenstette, J. J. (1999). [[pdf]](https://www.jair.org/media/613/live-613-1809-jair.pdf)

- **Machine Learning (chapter 13)** Mitchell T. (1997) [[web]](http://www.cs.cmu.edu/~tom/mlbook.html)

- **Reinforcement learning: An introduction.** Sutton, R. S., & Barto, A. G. (1998). Cambridge: MIT press. [[html]](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)

- **Reinforcement learning: An introduction (second edition).** Sutton, R. S., & Barto, A. G. (in progress). [[pdf]](http://incompleteideas.net/sutton/book/bookdraft2017june19.pdf)

- **Reinforcement Learning in a Nutshell.** Heidrich-Meisner, V., Lauer, M., Igel, C., & Riedmiller, M. A. (2007) [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9557&rep=rep1&type=pdf)

- **Statistical Reinforcement Learning: Modern Machine Learning Approaches**, Sugiyama, M. (2015) [[web]](https://www.crcpress.com/Statistical-Reinforcement-Learning-Modern-Machine-Learning-Approaches/Sugiyama/p/book/9781439856895)


License
--------
The MIT License (MIT)
Copyright (c) 2017 Massimiliano Patacchiola
Website: http://mpatacchiola.github.io/blog

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
