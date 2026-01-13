import numpy as np
import pandas as pd

def generate_ipl_data_raw(n_samples=1000):
    np.random.seed(42)

    teams = [
        'Mumbai Indians', 'Chennai Super Kings',
        'Royal Challengers Bangalore', 'Kolkata Knight Riders',
        'Delhi Capitals', 'Punjab Kings',
        'Rajasthan Royals', 'Sunrisers Hyderabad'
    ]

    venues = [
        'Wankhede Stadium', 'MA Chidambaram Stadium',
        'Eden Gardens', 'M. Chinnaswamy Stadium',
        'Arun Jaitley Stadium', 'Narendra Modi Stadium'
    ]

    data = []

    for _ in range(n_samples):
        team1 = np.random.choice(teams)
        team2 = np.random.choice([t for t in teams if t != team1])
        venue = np.random.choice(venues)

        team1_wins_last_5 = np.random.randint(0, 6)
        team2_wins_last_5 = np.random.randint(0, 6)

        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['bat', 'field'])

        home_advantage = int(
            ('Mumbai' in team1 and venue == 'Wankhede Stadium') or
            ('Chennai' in team1 and venue == 'MA Chidambaram Stadium') or
            ('Kolkata' in team1 and venue == 'Eden Gardens')
        )

        win_prob = 0.5
        win_prob += (team1_wins_last_5 - team2_wins_last_5) * 0.05
        win_prob += home_advantage * 0.15
        if toss_winner == team1:
            win_prob += 0.1

        winner = team1 if np.random.rand() < win_prob else team2

        data.append({
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "team1_wins_last_5": team1_wins_last_5,
            "team2_wins_last_5": team2_wins_last_5,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "home_advantage": home_advantage,
            "winner": winner
        })

    return pd.DataFrame(data)
