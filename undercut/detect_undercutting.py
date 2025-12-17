import pandas as pd
import shapely.geometry as sg
import numpy as np

def read_defender_path_data():
    return pd.read_parquet('data/defender_path_data.parquet')

def read_receiver_path_data():
    return pd.read_parquet('data/receiver_path_data.parquet')

def read_supplementary_data():
    return pd.read_csv('data/supplementary_data.csv')

def read_last_input_frame_data():
    return pd.read_parquet('data/coop-undercutting-last-input-frame-data.parquet')

def make_line_from_np(np_array):
    if np_array is None or len(np_array) < 2:
        return None
    return sg.LineString(np_array)

def detect_crossing(defender_line, receiver_line):
    return defender_line.intersects(receiver_line)

def create_movement_vector(previous_position, current_position):
    """
    Create a movement vector from the previous position to the current position.
    """
    return np.array([current_position[0] - previous_position[0], current_position[1] - previous_position[1]])

def calculate_distance_to_line_segment(point_x, point_y, line_start_x, line_start_y, line_end_x, line_end_y):
    """
    Calculate the distance from a point to a line segment using projection.
    """
    dx = line_end_x - line_start_x
    dy = line_end_y - line_start_y
    
    # Calculate the squared length of the line segment
    line_length_sq = dx * dx + dy * dy
    
    # Calculate parameter t: projection of defender position onto line
    if line_length_sq > 1e-10:  # threshold for numerical stability
        t = ((point_x - line_start_x) * dx + (point_y - line_start_y) * dy) / line_length_sq
        t = np.clip(t, 0, 1)
    else:
        t = 0.0  # if line has zero length, use start point
    
    closest_x = line_start_x + t * dx
    closest_y = line_start_y + t * dy
    distance = np.sqrt((closest_x - point_x)**2 + (closest_y - point_y)**2)
    
    return distance

def is_facing_ball_landing(defender_x, defender_y, defender_dir, ball_land_x, ball_land_y, tolerance_degrees=25):
    """
    Check if defender's direction is within tolerance_degrees of the direction to ball landing location.
    """
    # Calculate angle from defender to ball landing location
    dx = ball_land_x - defender_x
    dy = ball_land_y - defender_y
    
    # Calculate angle in standard coordinates
    angle_to_ball_math = np.degrees(np.arctan2(dy, dx))
    
    # Normalize to 0-360
    if angle_to_ball_math < 0:
        angle_to_ball_math += 360
    
    # Convert from math coordinates to NFL coordinates
    angle_to_ball_dir = (90 - angle_to_ball_math) % 360
    
    # Normalize to 0-360
    defender_dir_normalized = defender_dir % 360
    
    # Calculate the absolute difference
    angle_diff = abs(angle_to_ball_dir - defender_dir_normalized)
    
    # Handle wrap-around
    angle_diff = min(angle_diff, 360 - angle_diff)
    
    return angle_diff <= tolerance_degrees

def is_undercutting(movement_vector, has_crossed, distance_to_line_decreasing=None, facing_ball_landing=None, consecutive_frames_count=0):
    """
    Determine if the player is undercutting the route.
    
    Args:
        movement_vector: movement vector from previous to current position
        has_crossed: whether defender has crossed receiver's path
        distance_to_line_decreasing: whether distance to receiver-ball line is decreasing
        facing_ball_landing: whether defender is facing ball landing location
        consecutive_frames_count: number of consecutive frames with post_throw_downhill and lane_attack_signal
    
    Returns:
        bool: True if undercutting is detected
    """
    post_throw_downhill = movement_vector[0] < 0
    leaning_downhill_pre_throw = facing_ball_landing
    lane_attack_signal = distance_to_line_decreasing
    
    if has_crossed:
        return True
    if leaning_downhill_pre_throw and post_throw_downhill:
        return True
    return False

def evaluate_undercut(defender_path_data, receiver_path_data, last_input_frame_data, 
                     min_movement_towards_los=-0.1, min_distance_decrease=0.1):
    """
    Evaluate if undercutting occurs for a single play.
    Returns:
        bool: True if undercutting is detected, False otherwise
    """
    if len(last_input_frame_data) == 0:
        return False
    
    ball_land_x = last_input_frame_data['ball_land_x'].values[0]
    ball_land_y = last_input_frame_data['ball_land_y'].values[0]
    defender_last_x = last_input_frame_data['x'].values[0]
    defender_last_y = last_input_frame_data['y'].values[0]
    defender_dir = last_input_frame_data['dir'].values[0]
    
    # Check if defender is facing ball landing location
    facing_ball_landing = is_facing_ball_landing(
        defender_last_x, defender_last_y,
        defender_dir,
        ball_land_x, ball_land_y
    )
    
    # Make a copy of receiver_path_data and sort by source and frame_id
    sorted_receiver_path = receiver_path_data.copy()
    if 'source' in sorted_receiver_path.columns:
        sorted_receiver_path = sorted_receiver_path.sort_values(by=['source', 'frame_id'], ascending=[True, True])
    else:
        sorted_receiver_path = sorted_receiver_path.sort_values(by='frame_id')
    
    # Get receiver's last frame position
    if len(sorted_receiver_path) == 0:
        return False
    
    receiver_last_frame = sorted_receiver_path.iloc[-1]
    receiver_last_x = receiver_last_frame['x']
    receiver_last_y = receiver_last_frame['y']
    
    sorted_defender_path = defender_path_data.copy().sort_values(by='frame_id')
    
    if len(sorted_defender_path) < 2:
        return False
    
    receiver_line = make_line_from_np(sorted_receiver_path[['x', 'y']].to_numpy())
    if receiver_line is None:
        return False

    # Extract the first point from the sorted defender path
    first_defender_row = sorted_defender_path.head(1)[['x', 'y']]
    defender_points_list = first_defender_row.values.tolist()

    previous_position = None
    previous_distance_to_line = None
    has_crossed = False
    movement_vector = None
    consecutive_frames_count = 0
    undercutting_detected = False
    
    for index, row in sorted_defender_path.iloc[1:].iterrows():
        # Add the new point to the list
        current_position = (row['x'], row['y'])
        
        # Calculate distance from defender's current position to Jump Line
        current_distance_to_line = calculate_distance_to_line_segment(
            row['x'], row['y'],
            receiver_last_x, receiver_last_y,
            ball_land_x, ball_land_y
        )
        
        # Determine if distance to Jump Line is decreasing
        distance_to_line_decreasing = None
        if previous_distance_to_line is not None:
            distance_to_line_decreasing = current_distance_to_line < previous_distance_to_line
        
        if previous_position is not None:
            movement_vector = create_movement_vector(previous_position, current_position)
        
        previous_position = current_position
        previous_distance_to_line = current_distance_to_line
        defender_points_list.append(current_position)
        # Create a new LineString from all accumulated points
        defender_initial_line = make_line_from_np(defender_points_list)
        if defender_initial_line is not None and detect_crossing(defender_initial_line, receiver_line):
            has_crossed = True
            undercutting_detected = True
            break

        # Only check undercutting if we have a movement vector (need at least 2 positions)
        if movement_vector is not None:
            # Check if both post_throw_downhill and lane_attack_signal are true
            # post_throw_downhill: movement towards LOS must be negative AND below threshold
            post_throw_downhill = movement_vector[0] < min_movement_towards_los
            
            # lane_attack_signal: distance to Jump Line must be decreasing AND by at least the threshold
            distance_decrease_amount = None
            if previous_distance_to_line is not None and current_distance_to_line is not None:
                distance_decrease_amount = previous_distance_to_line - current_distance_to_line
                lane_attack_signal = (distance_decrease_amount >= min_distance_decrease)
            else:
                lane_attack_signal = False
            
            # If both conditions are true, increment counter; otherwise reset
            if post_throw_downhill and lane_attack_signal:
                consecutive_frames_count += 1
            else:
                consecutive_frames_count = 0
            
            # Check if we've had 3 consecutive frames or if undercutting was already detected
            # If undercutting is detected, break
            if undercutting_detected or (consecutive_frames_count >= 3):
                if not undercutting_detected:
                    undercutting_detected = True
                    break
            elif is_undercutting(movement_vector, has_crossed, distance_to_line_decreasing, facing_ball_landing, consecutive_frames_count):
                undercutting_detected = True
                break
    
    return undercutting_detected


def main():
    print("=" * 60)
    print("Full Undercutting Detection")
    print("=" * 60)
    
    print("Loading data...")
    defender_path_data = read_defender_path_data()
    receiver_path_data = read_receiver_path_data()
    last_input_frame_data = read_last_input_frame_data()
    
    print(f"Defender path data: {len(defender_path_data)} rows")
    print(f"Receiver path data: {len(receiver_path_data)} rows")
    print(f"Last input frame data: {len(last_input_frame_data)} rows")
    
    # Get unique plays from each dataset
    defender_plays = defender_path_data[['game_id', 'play_id']].drop_duplicates()
    receiver_plays = receiver_path_data[['game_id', 'play_id']].drop_duplicates()
    last_input_plays = last_input_frame_data[['game_id', 'play_id']].drop_duplicates()
    
    print(f"\nUnique plays in defender path data: {len(defender_plays)}")
    print(f"Unique plays in receiver path data: {len(receiver_plays)}")
    print(f"Unique plays in last input frame data: {len(last_input_plays)}")
    
    # Find intersection with all usable plays
    common_plays = defender_plays.merge(
        receiver_plays,
        on=['game_id', 'play_id'],
        how='inner'
    ).merge(
        last_input_plays,
        on=['game_id', 'play_id'],
        how='inner'
    )
    
    print(f"Common plays in all three datasets: {len(common_plays)}")
    print(f"\nProcessing {len(common_plays)} plays with complete data...")
    
    results = []
    
    for idx, play_row in common_plays.iterrows():
        game_id = play_row['game_id']
        play_id = play_row['play_id']
        
        # Filter data for this play
        play_defender_path = defender_path_data[
            (defender_path_data['game_id'] == game_id) & 
            (defender_path_data['play_id'] == play_id)
        ]
        play_receiver_path = receiver_path_data[
            (receiver_path_data['game_id'] == game_id) & 
            (receiver_path_data['play_id'] == play_id)
        ]
        play_last_input_frame = last_input_frame_data[
            (last_input_frame_data['game_id'] == game_id) & 
            (last_input_frame_data['play_id'] == play_id)
        ]
        
        # Get nfl_id from last_input_frame_data
        if len(play_last_input_frame) > 0:
            nfl_id = play_last_input_frame['nfl_id'].values[0] if 'nfl_id' in play_last_input_frame.columns else None
        else:
            nfl_id = None
        
        # Evaluate undercutting for this play
        is_undercutting_result = evaluate_undercut(
            play_defender_path, 
            play_receiver_path, 
            play_last_input_frame,
            min_movement_towards_los=-0.1,  # Must move at least 0.1 yards towards LOS per frame
            min_distance_decrease=0.1  # Distance must decrease by at least 0.1 yards per frame
        )
        
        results.append({
            'game_id': game_id,
            'play_id': play_id,
            'nfl_id': nfl_id,
            'is_undercutting': is_undercutting_result
        })
        
        # Print progress every 100 plays
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(common_plays)} plays...")
    
    # Create results df
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = 'undercutting_results.csv'
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Total plays processed: {len(results_df)}")
    print(f"Undercutting detected: {results_df['is_undercutting'].sum()}")
    print(f"Undercutting rate: {results_df['is_undercutting'].mean() * 100:.2f}%")
    print(f"\nResults saved to: {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

