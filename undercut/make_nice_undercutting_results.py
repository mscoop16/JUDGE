import pandas as pd

def read_supplementary_data():
    return pd.read_csv('data/supplementary_data.csv')

def read_undercutting_data():
    return pd.read_csv('undercutting_results.csv')

def read_model_predictions():
    return pd.read_csv('xgb_predictions.csv')

def read_players_data():
    return pd.read_parquet('all_players.parquet')

def main():
    supplementary_data = read_supplementary_data()
    undercutting_data = read_undercutting_data()
    model_predictions = read_model_predictions()
    players_data = read_players_data()
    # print(supplementary_data.head())
    # print(undercutting_data.head())
    # print(model_predictions.head())

    merged = model_predictions.merge(undercutting_data[['game_id', 'play_id', 'nfl_id', 'is_undercutting']], on=['game_id', 'play_id', 'nfl_id'], how='left')
    merged = merged.merge(supplementary_data[['game_id', 'play_id', 'team_coverage_type', 'team_coverage_man_zone', 'expected_points_added']], on=['game_id', 'play_id'], how='left')
    merged = merged.merge(players_data[['nfl_id', 'player_name', 'player_position']], on='nfl_id', how='left')
    print(merged.head())
    merged.to_csv('nice_undercutting_results.csv', index=False)
    
if __name__ == "__main__":
    main()