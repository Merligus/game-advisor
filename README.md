# Game Advisor

## Install

Install pytorch and other requirements.

- Change directory to `game-advisor/`;

- Create the environment in conda with python 3.11:
```bash
conda create -n GameAdvisor python=3.11 -y
conda activate GameAdvisor
```

```bash
pip install -r requirements.txt
```

- Create the env file with the following vars:
```bash
GAMESPOT_API_KEY=
RAWG_API_KEY=
IGDB_CLIENT_SECRET=
IGDB_CLIENT_ID=
```

## Run

### Create the user game review dataset

In game-advisor folder run:

```bash
python source/scripts/create_user_review_dataset.py
```

### Create the game dataset

```bash
python source/scripts/create_game_dataset.py
```
