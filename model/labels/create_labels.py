import pandas as pd
import numpy as np

def read_label_data():
    return pd.read_parquet('data/reachability_label_data.parquet')

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
    
    # Update the dataframe with normalized values
    data = data.copy()
    data['x'] = x_norm.round(2)
    data['y'] = y_centered.round(2)
    data['ball_land_x'] = ball_land_x_norm.round(2)
    data['ball_land_y'] = ball_land_y_centered.round(2)
    
    # Drop the temporary play constant columns
    data = data.drop(columns=['absolute_yardline_number_play', 'play_direction_play'])
    
    return data

def create_labels(defender_data, receiver_data):
    """
    Create labels by:
    1. Normalize coordinates for consistency with feature engineering
    2. Calculate line parameters from receiver's last frame position to ball landing
    3. Calculate the distance from each defender's last frame position to the line
    """
    # Normalize coordinates
    defender_data = normalize_coordinates(defender_data.copy())
    receiver_data = normalize_coordinates(receiver_data.copy())
    
    # Get receiver's last frame
    receiver_idx = receiver_data.groupby(['game_id', 'play_id'])['frame_id'].idxmax()
    receiver_last_frame = receiver_data.loc[receiver_idx].reset_index(drop=True)
    
    # Calculate distance to ball landing
    receiver_to_ball_distance = np.sqrt(
        (receiver_last_frame['ball_land_x'] - receiver_last_frame['x'])**2 + 
        (receiver_last_frame['ball_land_y'] - receiver_last_frame['y'])**2
    )

    
    # Calculate how much further the receiver is from LOS than the ball
    ball_further_than_receiver = receiver_last_frame['ball_land_x'] - receiver_last_frame['x']
    
    # Filter out plays where ball landing is > 5 yards away from receiver
    initial_play_count = len(receiver_last_frame)
    valid_plays_distance = receiver_to_ball_distance <= 5.0
    
    # Filter out plays where ball is more than 1 yard behind receiver
    valid_plays_los = ball_further_than_receiver <= 1.0
    
    # Combine both filters
    valid_plays = valid_plays_distance & valid_plays_los
    receiver_last_frame = receiver_last_frame[valid_plays].reset_index(drop=True)
    dropped_play_count = initial_play_count - len(receiver_last_frame)
    
    # Count how many were dropped
    dropped_distance = (~valid_plays_distance).sum()
    dropped_los = (~valid_plays_los).sum()
    dropped_both = ((~valid_plays_distance) & (~valid_plays_los)).sum()
    
    print(f"Filtered out {dropped_play_count} plays total:")
    print(f"  - {dropped_distance} plays where ball landing distance > 5 yards from receiver's last frame")
    print(f"  - {dropped_los} plays where ball is > 1 yard behind receiver (further from LOS)")
    print(f"  - {dropped_both} plays dropped by both filters (counted in both above)")
    print(f"Remaining plays: {len(receiver_last_frame)} out of {initial_play_count}")
    
    # Get the valid game_id and play_id pairs and filter defender data
    valid_play_keys = receiver_last_frame[['game_id', 'play_id']].drop_duplicates()
    defender_data = defender_data.merge(
        valid_play_keys,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    # Get defender's last frame
    defender_idx = defender_data.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
    defender_last_frame = defender_data.loc[defender_idx].reset_index(drop=True)
    
    # Calculate line parameters from receiver's last frame to ball landing
    x1 = receiver_last_frame['x'].values
    y1 = receiver_last_frame['y'].values
    x2 = receiver_last_frame['ball_land_x'].values
    y2 = receiver_last_frame['ball_land_y'].values
    
    # Create line parameters dataframe
    line_params = pd.DataFrame({
        'game_id': receiver_last_frame['game_id'].values,
        'play_id': receiver_last_frame['play_id'].values,
         'line_start_x': x1,
         'line_start_y': y1,
         'line_end_x': x2,
         'line_end_y': y2,
    })
    
    # Merge line parameters with defender data
    defender_with_line = pd.merge(
        defender_last_frame,
        line_params[['game_id', 'play_id', 'line_start_x', 'line_start_y', 'line_end_x', 'line_end_y']],
        on=['game_id', 'play_id'],
        how='left'
    )
    
    dx = defender_with_line['line_end_x'] - defender_with_line['line_start_x']
    dy = defender_with_line['line_end_y'] - defender_with_line['line_start_y']
    
    # Calculate the squared length of the line
    line_length_sq = dx * dx + dy * dy
    
    # Calculate parameter t: projection of defender position onto line
    t = np.where(
        line_length_sq > 1e-10,  # threshold for numerical stability
        ((defender_with_line['x'] - defender_with_line['line_start_x']) * dx + 
         (defender_with_line['y'] - defender_with_line['line_start_y']) * dy) / line_length_sq,
        0.0  # if line has zero length, use start point
    )
    t = np.clip(t, 0, 1)
    closest_x = defender_with_line['line_start_x'] + t * dx
    closest_y = defender_with_line['line_start_y'] + t * dy
    distance = np.sqrt((closest_x - defender_with_line['x'])**2 + (closest_y - defender_with_line['y'])**2)
    defender_with_line['distance_to_line'] = distance
    
    result = defender_with_line.drop(
        columns=['line_start_x', 'line_start_y', 'line_end_x', 'line_end_y']
    )
    return result

if __name__ == "__main__":
    label_data = read_label_data()
    print(f"Total rows in label_data: {len(label_data)}")
    print(f"Player role distribution:\n{label_data['player_role'].value_counts()}")
    
    defender_data = label_data[label_data['player_role'] == 'Defensive Coverage'].copy()
    receiver_data = label_data[label_data['player_role'] == 'Targeted Receiver'].copy()
    
    print(f"\nDefender data rows: {len(defender_data)}")
    print(f"Receiver data rows: {len(receiver_data)}")
    
    df = create_labels(defender_data, receiver_data)
    
    print(f"\nResult dataframe shape: {df.shape}")
    print(f'Mean distance to line: {df["distance_to_line"].mean():.2f}')
    print(f'Median distance to line: {df["distance_to_line"].median():.2f}')
    print(f'Std distance to line: {df["distance_to_line"].std():.2f}')
    print(f'Min distance to line: {df["distance_to_line"].min():.2f}')
    max_row = df[df["distance_to_line"] == df["distance_to_line"].max()].iloc[0]
    print(f'Play with max distance to line: game_id={max_row["game_id"]}, play_id={max_row["play_id"]}, distance={max_row["distance_to_line"]:.2f}')
    min_row = df[df["distance_to_line"] == df["distance_to_line"].min()].iloc[0]
    print(f'Play with min distance to line: game_id={min_row["game_id"]}, play_id={min_row["play_id"]}, distance={min_row["distance_to_line"]:.2f}')
    print(f'Max distance to line: {df["distance_to_line"].max():.2f}')
    print(f"\nResult columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Save the result with 'distance_to_line' labels
    df[['game_id', 'play_id', 'nfl_id', 'distance_to_line']].to_parquet('data/line_labels_normalized.parquet', index=False)
    print('\nSaved to data/line_labels_normalized.parquet')