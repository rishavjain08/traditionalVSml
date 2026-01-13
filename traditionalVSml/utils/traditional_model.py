def predict_traditional(
    team1,
    team2,
    venue,
    toss_winner,
    team1_form,
    team2_form
):
    if venue == "Wankhede Stadium" and team1 == "Mumbai Indians":
        return team1, 0.75
    elif venue == "MA Chidambaram Stadium" and team2 == "Chennai Super Kings":
        return team2, 0.75
    elif team1_form >= 4 and team2_form <= 2:
        return team1, 0.65
    elif team2_form >= 4 and team1_form <= 2:
        return team2, 0.65
    elif toss_winner == team1:
        return team1, 0.55
    else:
        return team2, 0.55
