import pandas as pd
import glob
from pathlib import Path
import numpy as np

def normalize_coordinates(data):
    """
    Normalize coordinates and angles for all rows in the dataset.
    Normalizes based on play direction and line of scrimmage.
    """
    FIELD_LENGTH = 120
    FIELD_WIDTH = 53.3
    
    # Get play-level constants
    play_constants = data.groupby(['game_id', 'play_id']).agg({
        'absolute_yardline_number': 'first',
        'play_direction': 'first'
    }).reset_index()
    
    # Merge play constants
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
    
    # Update the df with normalized values
    data = data.copy()
    data['x'] = x_norm.round(2)
    data['y'] = y_centered.round(2)
    data['ball_land_x'] = ball_land_x_norm.round(2)
    data['ball_land_y'] = ball_land_y_centered.round(2)
    data['o'] = o_new.round(2)
    data['dir'] = dir_new.round(2)
    
    # Drop the temporary play constants
    data = data.drop(columns=['absolute_yardline_number_play', 'play_direction_play'])
    
    return data

def read_sumer_coverages_player_play():
    return pd.read_parquet('data/sumer_coverages_player_play.parquet')

def load_input_data():
    """
    Load all input files.
    """
    print("Loading input CSV files...")
    input_files = sorted(glob.glob('data/train/input_*.csv'))
    
    if not input_files:
        raise ValueError("No input*.csv files found in data/train/")
    
    print(f"Found {len(input_files)} input files")
    
    # Read all input files and combine
    input_dfs = []
    for file_path in input_files:
        print(f"  Reading {Path(file_path).name}...")
        df = pd.read_csv(file_path, low_memory=False)
        input_dfs.append(df)
    
    combined_input = pd.concat(input_dfs, ignore_index=True)
    print(f"Combined input data: {len(combined_input)} rows")
    
    return combined_input


def main():
    print("=" * 60)
    print("Input Frame Data Preprocessing")
    print("=" * 60)
    
    input_data = load_input_data()
    
    print("\nProcessing data...")
    
    # Get targeted defenders
    print("Loading targeted defenders data...")
    sumer_coverages_player_play = read_sumer_coverages_player_play()
    sumer_has_targeted_defender = sumer_coverages_player_play[sumer_coverages_player_play['targeted_defender'] == True]
    sumer_targeted_defenders = sumer_has_targeted_defender[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    # Ensure correct data types for joining
    sumer_targeted_defenders = sumer_targeted_defenders.astype({
        'game_id': 'int64',
        'play_id': 'int64',
        'nfl_id': 'int64'
    })
    print(f"Number of targeted defenders: {len(sumer_targeted_defenders)}")
    
    # Separate defenders and receivers from input data
    defender_data = input_data[
        input_data['player_role'] == 'Defensive Coverage'
    ].copy()
    receiver_data = input_data[
        input_data['player_role'] == 'Targeted Receiver'
    ].copy()
    
    print(f"Defender rows (before filtering): {len(defender_data)}")
    print(f"Receiver rows: {len(receiver_data)}")
    
    # Filter defenders to only include targeted defenders
    print("Filtering defenders to only targeted defenders...")
    defender_data = defender_data.merge(
        sumer_targeted_defenders,
        on=['game_id', 'play_id', 'nfl_id'],
        how='inner'
    )
    print(f"Defender rows (after filtering): {len(defender_data)}")
    
    # Normalize coordinates
    print("Normalizing coordinates...")
    defender_data = normalize_coordinates(defender_data)
    receiver_data = normalize_coordinates(receiver_data)
    
    # Get receiver's last pre-throw frame per play
    print("Getting receiver last input frames...")
    receiver_idx = receiver_data.groupby(['game_id', 'play_id'])['frame_id'].idxmax()
    receiver_last_frame = receiver_data.loc[receiver_idx].reset_index(drop=True)
    
    # Get defender's last pre-throw frame per play
    print("Getting defender last input frames...")
    # First get last frame per defender per play
    defender_idx = defender_data.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
    defender_last_frame_all = defender_data.loc[defender_idx].reset_index(drop=True)
    
    defender_last_frame = defender_last_frame_all.groupby(['game_id', 'play_id']).first().reset_index()
    
    # Select only the columns needed
    receiver_cols = ['game_id', 'play_id', 'x', 'y', 'dir', 'o', 's', 
                     'ball_land_x', 'ball_land_y', 'num_frames_output']
    receiver_cols = [col for col in receiver_cols if col in receiver_last_frame.columns]
    
    defender_cols = ['game_id', 'play_id', 'nfl_id', 'x', 'y', 'dir', 'o', 's']
    defender_cols = [col for col in defender_cols if col in defender_last_frame.columns]
    
    # Rename receiver columns with suffix
    receiver_selected = receiver_last_frame[receiver_cols].copy()
    receiver_selected = receiver_selected.rename(columns={
        'x': 'x_receiver',
        'y': 'y_receiver',
        'dir': 'dir_receiver',
        'o': 'o_receiver',
        's': 's_receiver'
    })
    
    defender_selected = defender_last_frame[defender_cols].copy()
    
    # Merge receiver and defender data
    print("Merging receiver and defender data...")
    play_data = pd.merge(
        defender_selected,
        receiver_selected,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    print(f"Total plays: {len(play_data)}")
    
    # Save results
    output_path = 'data/coop-undercutting-last-input-frame-data.parquet'
    print(f"\nSaving preprocessed data to {output_path}...")
    play_data.to_parquet(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total plays: {len(play_data)}")
    print(f"Columns: {list(play_data.columns)}")
    print("\nDone!")


if __name__ == "__main__":
    main()

