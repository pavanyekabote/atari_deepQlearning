{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Conv2D, Flatten\n",
    "from keras.optimizers import Adam\n",
    "from keras.preprocessing import image\n",
    "import gym\n",
    "import numpy as np\n",
    "from collections import deque\n",
    "import random\n",
    "import sys\n",
    "import pandas as pd\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 257,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(Box(210, 160, 3), Discrete(4))"
      ]
     },
     "execution_count": 257,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "env = gym.make('Breakout-v0')\n",
    "env.observation_space, env.action_space"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DQN:\n",
    "    \n",
    "    def __init__(self, state_size, action_size):\n",
    "        self.memory = deque(maxlen=2000)\n",
    "        self.state_size = state_size\n",
    "        self.action_size = action_size\n",
    "        self.epsilon= 1\n",
    "        self.eps_decay_rate = 0.9995\n",
    "        self.gamma = 0.70\n",
    "        self.lr = 0.01\n",
    "        self.min_eps = 0.01\n",
    "        self.model = self._build_model()\n",
    "    \n",
    "    def preprocess(self, image):\n",
    "        img = Image.fromarray(image)\n",
    "        state =  np.expand_dims(np.asarray(np.max(img.resize((84,84)), axis=2)),axis=2)\n",
    "        ns = np.expand_dims(state,axis=0)\n",
    "        #print(\"State DIMS: \",state.shape, \" NS DIMES: \",ns.shape)\n",
    "        return ns\n",
    "        \n",
    "    \n",
    "    def _build_model(self):\n",
    "        model = Sequential()\n",
    "        model.add(Conv2D(32, 8 , strides=(2, 2), padding='valid', activation='relu', input_shape=self.state_size))\n",
    "        model.add(Conv2D(64, 4 , strides=(4, 4), padding='valid', activation='relu'))\n",
    "        model.add(Conv2D(128, 3 ,strides=(2, 2), padding='valid', activation='relu'))\n",
    "        model.add(Flatten())\n",
    "        model.add(Dense(512, activation='relu'))\n",
    "        model.add(Dense(self.action_size, activation='linear'))\n",
    "        model.compile(loss='mse', optimizer=Adam(lr=self.lr), metrics=['accuracy'])\n",
    "        return model\n",
    "    \n",
    "    def mem_size(self):\n",
    "        return len(self.memory)\n",
    "    \n",
    "    def check_shape(self, state):\n",
    "        return True if state.shape[2] > self.state_size[2] else False\n",
    "        \n",
    "    def act(self, state):\n",
    "        #print(\"STATE SIZE: \", state.shape)\n",
    "        if self.check_shape(state):\n",
    "            state = self.preprocess(state)\n",
    "        #print(\"STATE AFTER \",state.shape)\n",
    "        prob = np.random.rand()\n",
    "        if prob < self.epsilon:\n",
    "            action = random.randrange(self.action_size)\n",
    "        else:\n",
    "            action = np.argmax(self.model.predict(state)[0])\n",
    "        return action\n",
    "            \n",
    "    def remember(self, state, action, reward, next_state, done):\n",
    "        if self.check_shape(state) or self.check_shape(next_state):\n",
    "            state = self.preprocess(state)\n",
    "            next_state = self.preprocess(next_state)\n",
    "        self.memory.append((state, action, reward, next_state, done))\n",
    "    \n",
    "    def replay(self, batch_size):\n",
    "        minibatch = random.sample(self.memory, batch_size)\n",
    "        for state, action, reward, next_state, done in minibatch:\n",
    "            target = reward\n",
    "            if not done:\n",
    "                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])\n",
    "            target_f =self.model.predict(state)\n",
    "            target_f[0][action] = target\n",
    "            self.model.fit(state, target_f, epochs=1, verbose=0)\n",
    "        if self.epsilon > self.min_eps:\n",
    "            self.epsilon *= self.eps_decay_rate\n",
    "            \n",
    "    def load(self, name):\n",
    "        self.model.load_weights(name)\n",
    "\n",
    "    def save(self, name):\n",
    "        self.model.save_weights(name)\n",
    "\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 276,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(env.observation_space.shape[:2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 277,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize state_size, action_size\n",
    "state_size = (84,84,1)\n",
    "action_size = env.action_space.n\n",
    "\n",
    "# Create an Agent with required State and Action sizes\n",
    "agent = DQN(state_size, action_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(env, n_episodes=100, batch_size=32):\n",
    "    \n",
    "    avg_reward = []\n",
    "    max_avg = -np.inf\n",
    "    for i_ep in range(n_episodes):\n",
    "        \n",
    "        state = env.reset()\n",
    "        total_reward = 0\n",
    "        for t in range(600):\n",
    "            action = agent.act(state)\n",
    "            next_state, reward, done, _ = env.step(action)\n",
    "            total_reward += reward\n",
    "            #env.render()\n",
    "            agent.remember(state, action, reward, next_state, done)\n",
    "            if done:\n",
    "                break\n",
    "        avg_reward.append(total_reward)\n",
    "        avg = np.mean(avg_reward[-100:])\n",
    "        if (max_avg < avg):\n",
    "            max_avg = avg\n",
    "        print(\"\\repisode: {}/{},Max: {:.2f} score: {:.2f}\".format(i_ep+1, n_episodes, max_avg, avg),end=\"\")\n",
    "        sys.stdout.flush()\n",
    "        \n",
    "        if agent.mem_size() >= batch_size:\n",
    "            agent.replay(batch_size)\n",
    "    return avg_reward"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 279,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "episode: 100/100,Max: 1.67 score: 1.36"
     ]
    }
   ],
   "source": [
    "scores = train(env, n_episodes=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test(env, n_episodes=10, batch_size=32):\n",
    "    \n",
    "    avg_reward = []\n",
    "    max_avg = -np.inf\n",
    "    for i_ep in range(n_episodes):\n",
    "        \n",
    "        state = env.reset()\n",
    "        total_reward = 0\n",
    "        done = False\n",
    "        while True:\n",
    "            action = agent.act(state)\n",
    "            next_state, reward, done, _ = env.step(action)\n",
    "            env.render()\n",
    "            total_reward += reward\n",
    "            agent.remember(state, action, reward, next_state, done)\n",
    "            if done:\n",
    "                break\n",
    "        env.close()\n",
    "        avg_reward.append(total_reward)\n",
    "        avg = np.mean(avg_reward)\n",
    "        if (max_avg < avg):\n",
    "            max_avg = avg\n",
    "        print(\"\\repisode: {}/{},Max: {:.2f} score: {:.2f}\".format(i_ep+1, n_episodes, max_avg, avg),end=\"\")\n",
    "        sys.stdout.flush()\n",
    "        \n",
    "    return avg_reward"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "episode: 10/10,Max: 1.12 score: 0.90"
     ]
    }
   ],
   "source": [
    "scores_test = test(env)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "metadata": {},
   "outputs": [],
   "source": [
    "state = env.reset()\n",
    "\n",
    "done = False\n",
    "while True:\n",
    "    next_state, reward, done, _ = env.step(env.action_space.sample())\n",
    "    env.render()\n",
    "    if done:\n",
    "        break\n",
    "env.close()\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_scores(scores, rolling_window=100):\n",
    "    \"\"\"Plot scores and optional rolling mean using specified window.\"\"\"\n",
    "    plt.plot(scores); plt.title(\"Scores\");\n",
    "    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()\n",
    "    plt.plot(rolling_mean);\n",
    "    return rolling_mean\n",
    "\n",
    "rolling_mean = plot_scores(scores_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 250,
   "metadata": {},
   "outputs": [],
   "source": [
    "env.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
