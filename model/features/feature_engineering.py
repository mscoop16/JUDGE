import pandas as pd
import numpy as np

def read_pre_data():
    pre_data = pd.read_parquet('data/underneath_input_tracking_data.parquet')
    return pre_data

def read_sumer_data():
    sumer_data = pd.read_parquet('data/sumer_coverages_player_play.parquet')[['game_id', 'play_id', 'nfl_id', 'coverage_responsibility', 'alignment']]

    desired_sumer_data = sumer_data[~sumer_data['alignment'].isin(['EDGE', 'DT'])]
    desired_sumer_data = desired_sumer_data[~desired_sumer_data['coverage_responsibility'].isin(['PREVENT'])]

    return desired_sumer_data

def calculate_player_features(pre_data):
    grouped = pre_data.groupby(['nfl_id']).agg(
        player_average_speed=('s', 'mean'),
        player_max_speed=('s', 'max'),
        player_average_acceleration=('a', 'mean'),
        player_max_acceleration=('a', 'max')
    ).reset_index()
    grouped['player_average_speed'] = grouped['player_average_speed'].round(2)
    grouped['player_average_acceleration'] = grouped['player_average_acceleration'].round(2)
    grouped['player_max_speed'] = grouped['player_max_speed'].round(2)
    grouped['player_max_acceleration'] = grouped['player_max_acceleration'].round(2)
    return grouped

def normalize_coordinates(data):
    """
    Normalize coordinates and angles for all rows in the dataset.
    Normalizes based on play direction and line of scrimmage.
    """
    FIELD_LENGTH = 120
    FIELD_WIDTH = 53.3
    
    # Get play-level constants (line of scrimmage and play direction)
    play_constants = data.groupby(['game_id', 'play_id']).agg({
        'absolute_yardline_number': 'first',
        'play_direction': 'first'
    }).reset_index()
    
    # Merge play constants back to all rows
    data = data.merge(play_constants, on=['game_id', 'play_id'], suffixes=('', '_play'))
    line_of_scrimmage = data['absolute_yardline_number_play']
    play_direction = data['play_direction_play']
    
    is_left_play = play_direction == 'left'
    
    # Normalize x coordinates
    ball_land_x_new = np.where(is_left_play, FIELD_LENGTH - data['ball_land_x'], data['ball_land_x'])
    x_new = np.where(is_left_play, FIELD_LENGTH - data['x'], data['x'])
    los_new = np.where(is_left_play, FIELD_LENGTH - line_of_scrimmage, line_of_scrimmage)
    x_norm = x_new - los_new
    ball_land_x_norm = ball_land_x_new - los_new
    
    # Normalize y coordinates
    y_new = np.where(is_left_play, FIELD_WIDTH - data['y'], data['y'])
    y_centered = y_new - (FIELD_WIDTH / 2)
    ball_land_y_new = np.where(is_left_play, FIELD_WIDTH - data['ball_land_y'], data['ball_land_y'])
    ball_land_y_centered = ball_land_y_new - (FIELD_WIDTH / 2)
    
    # Adjust angles
    angle_adjustment = np.where(is_left_play, 180, 0)
    o_new = (data['o'] + angle_adjustment) % 360
    dir_new = (data['dir'] + angle_adjustment) % 360
    
    # Update the dataframe with normalized values
    data = data.copy()
    data['x'] = x_norm.round(2)
    data['y'] = y_centered.round(2)
    data['o'] = o_new.round(2)
    data['dir'] = dir_new.round(2)
    data['ball_land_x'] = ball_land_x_norm.round(2)
    data['ball_land_y'] = ball_land_y_centered.round(2)
    
    # Drop the temporary play constant columns
    data = data.drop(columns=['absolute_yardline_number_play', 'play_direction_play'])
    
    return data

def calculate_last_frame_features(pre_data):
    """
    Extract features from the last frame for each player in each play.
    """
    idx = pre_data.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
    last_frame = pre_data.loc[idx].reset_index(drop=True)
    
    columns_to_include = ['game_id', 'play_id', 'nfl_id', 'x', 'y', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y']
    last_frame = last_frame[columns_to_include]
    last_frame['dist_to_ball_land'] = np.sqrt((last_frame['ball_land_x'] - last_frame['x'])**2 + (last_frame['ball_land_y'] - last_frame['y'])**2)
    return last_frame

def calculate_directional_alignment_feature(pre_data):
    """
    Calculate the directional alignment feature for each player in each play.
    """
    # Get only the last frame for each player in each play
    idx = pre_data.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
    last_frame = pre_data.loc[idx].reset_index(drop=True)
    
    df = last_frame.copy()
    dx = df['ball_land_x'] - df['x']
    dy = df['ball_land_y'] - df['y']
    target_angle = np.arctan2(dy, dx)
    dir_rad = np.deg2rad(df['dir'])
    df['dir_alignment'] = np.cos(dir_rad - target_angle)
    return df[['game_id', 'play_id', 'nfl_id', 'dir_alignment']]

def combine_features(player_features, last_frame_features, coverage_features, directional_alignment_feature):
    # Merge last_frame_features and coverage_features
    combined_features = pd.merge(last_frame_features, coverage_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    # Merge player_features
    combined_features = pd.merge(combined_features, player_features, on=['nfl_id'], how='left')
    # Merge directional_alignment_feature
    combined_features = pd.merge(combined_features, directional_alignment_feature, on=['game_id', 'play_id', 'nfl_id'], how='left')
    return combined_features

if __name__ == "__main__":
    pre_data = read_pre_data()
    # Normalize coordinates once for all feature calculations
    pre_data_normalized = normalize_coordinates(pre_data)
    
    print('Calculating player features...')
    player_features = calculate_player_features(pre_data_normalized)
    print('Calculating last frame features...')
    last_frame_features = calculate_last_frame_features(pre_data_normalized)
    print('Reading coverage features...')
    coverage_features = read_sumer_data()
    print('Calculating directional alignment feature...')
    directional_alignment_feature = calculate_directional_alignment_feature(pre_data_normalized)
    print('Combining features...')
    combined_features = combine_features(player_features, last_frame_features, coverage_features, directional_alignment_feature)
    print('Saving features...')
    print(combined_features.head())
    print(combined_features.columns)
    combined_features.to_parquet('data/reachability-all-features-normalized.parquet', index=False)
    print('Saved to data/reachability-all-features-normalized.parquet')