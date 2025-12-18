import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from shapely.geometry import LineString
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def load_before_data():
    df = pd.read_parquet('data/train/combined_filtered.parquet')
    df = df[df['data_type'] == 'input'].copy()
    return df

def load_after_data():
    df = pd.read_parquet('data/train/combined_filtered.parquet')
    df = df[df['data_type'] == 'output'].copy()
    return df

def load_supplementary_data():
    df = pd.read_csv('data/supplementary_data.csv', low_memory=False)
    return df

_xgb_predictions_cache = None

def load_xgb_predictions():
    global _xgb_predictions_cache
    if _xgb_predictions_cache is None:
        _xgb_predictions_cache = pd.read_csv('xgb_predictions.csv')
    return _xgb_predictions_cache

def get_expected_distance(game_id, play_id):
    df = load_xgb_predictions()
    match = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)]
    if not match.empty:
        return match['predicted'].iloc[0]
    return None

def animate_play(before_data, after_data, supplementary_row=None, save_path='play_animation.gif', expected_distance=None):
    """
    Animate play: pre-throw (fast), post-throw (slow) with distance graph, jump zone, and jump line
    """

    # Filter players
    defender_before = before_data[before_data['player_role'] == 'Defensive Coverage'].copy()
    receiver_before = before_data[before_data['player_role'] == 'Targeted Receiver'].copy()
    defender_nfl_id = defender_before['nfl_id'].iloc[0]
    receiver_nfl_id = receiver_before['nfl_id'].iloc[0]

    defender_after = after_data[after_data['nfl_id'] == defender_nfl_id].copy()
    receiver_after = after_data[after_data['nfl_id'] == receiver_nfl_id].copy()

    defender_before = defender_before.sort_values('frame_id')
    receiver_before = receiver_before.sort_values('frame_id')
    defender_after = defender_after.sort_values('frame_id')
    receiver_after = receiver_after.sort_values('frame_id')

    # Ball landing
    ball_land_x = before_data['ball_land_x'].iloc[0] if 'ball_land_x' in before_data.columns else None
    ball_land_y = before_data['ball_land_y'].iloc[0] if 'ball_land_y' in before_data.columns else None

    # Prepare defender distance to line
    if ball_land_x is not None and ball_land_y is not None:
        line_start_x = receiver_after['x'].iloc[-1]
        line_start_y = receiver_after['y'].iloc[-1]
        dx = ball_land_x - line_start_x
        dy = ball_land_y - line_start_y
        line_length_sq = dx*dx + dy*dy
        distances_to_line = []
        for _, row in defender_after.iterrows():
            t = ((row['x'] - line_start_x)*dx + (row['y'] - line_start_y)*dy)/line_length_sq if line_length_sq>1e-10 else 0
            t = np.clip(t,0,1)
            closest_x = line_start_x + t*dx
            closest_y = line_start_y + t*dy
            distance = np.sqrt((closest_x - row['x'])**2 + (closest_y - row['y'])**2)
            distances_to_line.append(distance)
        defender_after['distance_to_line'] = distances_to_line
    else:
        defender_after['distance_to_line'] = 0

    # Calculate view bounds with padding
    all_x = list(defender_before['x']) + list(receiver_before['x']) + list(defender_after['x']) + list(receiver_after['x'])
    all_y = list(defender_before['y']) + list(receiver_before['y']) + list(defender_after['y']) + list(receiver_after['y'])
    if ball_land_x is not None and ball_land_y is not None:
        all_x.append(ball_land_x)
        all_y.append(ball_land_y)
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add padding to prevent lines from going off the page
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.3  # 30%
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    fig = plt.figure(figsize=(24, 10), facecolor='white')
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    # Draw field background
    ax_left.add_patch(plt.Rectangle((y_min, x_min), y_max - y_min, x_max - x_min, 
                                   facecolor='#2d5016', alpha=0.3, zorder=0))
    
    # Draw yard lines
    for yard in range(int(x_min), int(x_max) + 1, 5):
        if x_min <= yard <= x_max:
            ax_left.axhline(yard, color='white', linewidth=0.5, alpha=0.2, linestyle='--', zorder=0)
    for yard in range(int(y_min), int(y_max) + 1, 5):
        if y_min <= yard <= y_max:
            ax_left.axvline(yard, color='white', linewidth=0.5, alpha=0.2, linestyle='--', zorder=0)

    # Initialize pre-throw lines
    def_before_line, = ax_left.plot([], [], color='#4169E1', linestyle='--', linewidth=3, 
                                    alpha=0.8, zorder=2)
    rec_before_line, = ax_left.plot([], [], color='#DC143C', linestyle='--', linewidth=3, 
                                    alpha=0.8, zorder=2)
    
    # Initialize post-throw lines
    def_after_line, = ax_left.plot([], [], color='#0000CD', linestyle='-', linewidth=3.5, 
                                   alpha=0.9, zorder=3)
    rec_after_line, = ax_left.plot([], [], color='#8B0000', linestyle='-', linewidth=3.5, 
                                   alpha=0.9, zorder=3)
    
    # Jump line
    jump_line, = ax_left.plot([], [], color='#32CD32', linestyle='-', linewidth=4, 
                                         alpha=0.8, zorder=1, visible=False)
    
    if expected_distance is None:
        print("Expected distance is not provided")
        return

    # Jump zone area
    jump_zone_area = None
    if ball_land_x is not None and ball_land_y is not None:
        jump_line = LineString([
            (receiver_after['x'].iloc[-1], receiver_after['y'].iloc[-1]),
            (ball_land_x, ball_land_y)
        ])
        jump_zone = jump_line.buffer(expected_distance)
        zx, zy = jump_zone.exterior.xy
        jump_zone_area = ax_left.fill(
            zy, zx,
            color='crimson',
            alpha=0.15,
            zorder=1
        )[0]

    # Moving player markers
    def_marker = ax_left.scatter([], [], s=250, c='#4169E1', edgecolors='#000080', 
                                linewidths=2.5, zorder=8, marker='o', alpha=0.95)
    rec_marker = ax_left.scatter([], [], s=250, c='#DC143C', edgecolors='#8B0000', 
                                linewidths=2.5, zorder=8, marker='o', alpha=0.95)
    
    # Ball landing marker
    if ball_land_x is not None and ball_land_y is not None:
        ball_marker = ax_left.scatter(ball_land_y, ball_land_x, s=400, marker='*', color='#FFD700', 
                                     edgecolors='#FF8C00', linewidths=2.5, zorder=6, alpha=0.9)
    else:
        ball_marker = None

    # Set up axes
    ax_left.set_xlim(y_min, y_max)
    ax_left.set_ylim(x_min, x_max)
    ax_left.set_xlabel('Field Width (yards)', fontsize=12, fontweight='bold', color='#333333')
    ax_left.set_ylabel('Field Length (yards)', fontsize=12, fontweight='bold', color='#333333')
    ax_left.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.set_facecolor('#f8f8f8')
    ax_left.tick_params(colors='#333333', labelsize=10)
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='#DC143C', linewidth=3.5, label='Receiver'),
        Line2D([0], [0], color='#4169E1', linewidth=3.5, label='Defender'),
    ]
    if jump_zone_area is not None:
        legend_elements.append(Patch(facecolor='crimson', alpha=0.15, label='Expected Distance Zone'))
    
    legend_left = ax_left.legend(handles=legend_elements, loc='lower right', fontsize=14, 
                                framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    legend_left.get_frame().set_facecolor('white')

    # Title
    if supplementary_row is not None:
        title_parts = []
        if 'week' in supplementary_row and pd.notna(supplementary_row['week']):
            title_parts.append(f"Week {supplementary_row['week']}")
        if 'play_description' in supplementary_row and pd.notna(supplementary_row['play_description']):
            if title_parts:
                title_parts.append(f" - {supplementary_row['play_description'][:60]}...")
            else:
                title_parts.append(f"{supplementary_row['play_description'][:60]}...")
        if 'route_of_targeted_receiver' in supplementary_row and pd.notna(supplementary_row['route_of_targeted_receiver']):
            title_parts.append(f"\nRoute: {supplementary_row['route_of_targeted_receiver']}")
        if 'pass_result' in supplementary_row and pd.notna(supplementary_row['pass_result']):
            title_parts.append(f" | Result: {supplementary_row['pass_result']}")
        ax_left.set_title("".join(title_parts), fontsize=18, fontweight='bold', pad=5, color='#1a1a1a')
    else:
        ax_left.set_title("Player Trajectories (Pre-throw & Post-throw)", fontsize=18, fontweight='bold', pad=5, color='#1a1a1a')

    # Right plot: defender distance line
    distance_line_plot, = ax_right.plot([], [], color='#32CD32', alpha=0.7, linewidth=2, marker='o', markersize=3, visible=False)
    expected_line = ax_right.axhline(y=expected_distance, color='crimson', linestyle='--', linewidth=1, visible=True)
    ax_right.set_xlabel('Frame ID', fontsize=12, fontweight='bold', color='#333333')
    ax_right.set_ylabel('Distance to Line (yards)', fontsize=12, fontweight='bold', color='#333333')
    ax_right.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax_right.set_facecolor('#f8f8f8')
    ax_right.tick_params(colors='#333333', labelsize=10)
    if not defender_after.empty:
        ax_right.set_xlim(defender_after['frame_id'].min(), defender_after['frame_id'].max())
        ax_right.set_ylim(0, max(defender_after['distance_to_line'].max()*1.2, 1.5))

    # Right plot legend
    legend_elements_right = [
        Line2D([0], [0], color='#32CD32', linewidth=2, marker='o', markersize=3, label='Distance to Line'),
        Line2D([0], [0], color='crimson', linestyle='--', linewidth=1, label='Expected Distance')
    ]
    legend_right = ax_right.legend(handles=legend_elements_right, loc='best', fontsize=14, 
                                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    legend_right.get_frame().set_facecolor('white')

    # Animation setup
    pre_throw_frames = min(len(defender_before), len(receiver_before))
    post_throw_frames = min(len(defender_after), len(receiver_after))
    
    # Total animation frames
    total_frames = pre_throw_frames + post_throw_frames

    def update(frame):
        if frame < pre_throw_frames:
            # Pre-throw phase
            current_frame = frame + 1
            def_before_line.set_data(defender_before['y'][:current_frame], defender_before['x'][:current_frame])
            rec_before_line.set_data(receiver_before['y'][:current_frame], receiver_before['x'][:current_frame])
            
            # Update moving player markers
            def_marker.set_offsets([[defender_before['y'].iloc[frame], defender_before['x'].iloc[frame]]])
            rec_marker.set_offsets([[receiver_before['y'].iloc[frame], receiver_before['x'].iloc[frame]]])
            
            # Hide post-throw lines
            def_after_line.set_data([], [])
            rec_after_line.set_data([], [])
            jump_line.set_data([], [])
            jump_line.set_visible(False)
            distance_line_plot.set_visible(False)
            expected_line.set_visible(False)
            if jump_zone_area is not None:
                jump_zone_area.set_alpha(0.0)
            
            artists = [def_before_line, rec_before_line, def_after_line, rec_after_line,
                      jump_line, distance_line_plot, expected_line,
                      def_marker, rec_marker]
            if jump_zone_area is not None:
                artists.append(jump_zone_area)
            return tuple(artists)
        else:
            # Post-throw phase
            post_frame = frame - pre_throw_frames
            current_frame = post_frame + 1
            
            # Show pre-throw lines completely
            def_before_line.set_data(defender_before['y'], defender_before['x'])
            rec_before_line.set_data(receiver_before['y'], receiver_before['x'])
            
            # Animate post-throw lines
            def_after_line.set_data(defender_after['y'][:current_frame], defender_after['x'][:current_frame])
            rec_after_line.set_data(receiver_after['y'][:current_frame], receiver_after['x'][:current_frame])
            
            # Update moving player markers
            def_marker.set_offsets([[defender_after['y'].iloc[post_frame], defender_after['x'].iloc[post_frame]]])
            rec_marker.set_offsets([[receiver_after['y'].iloc[post_frame], receiver_after['x'].iloc[post_frame]]])
            
            # Show post-throw lines
            if ball_land_x is not None and ball_land_y is not None:
                jump_line.set_data([receiver_after['y'].iloc[-1], ball_land_y],
                                               [receiver_after['x'].iloc[-1], ball_land_x])
                jump_line.set_visible(True)
            if jump_zone_area is not None:
                jump_zone_area.set_alpha(0.15)
            
            # Update distance graph
            distance_line_plot.set_data(defender_after['frame_id'][:current_frame], defender_after['distance_to_line'][:current_frame])
            distance_line_plot.set_visible(True)
            expected_line.set_visible(True)
            
            artists = [def_before_line, rec_before_line, def_after_line, rec_after_line,
                      jump_line, distance_line_plot, expected_line,
                      def_marker, rec_marker]
            if jump_zone_area is not None:
                artists.append(jump_zone_area)
            return tuple(artists)

    # Create animation with slower pre-throw and post-throw
    pre_throw_slowdown_factor = 3
    post_throw_slowdown_factor = 6
    effective_pre_frames = pre_throw_frames * pre_throw_slowdown_factor
    effective_post_frames = post_throw_frames * post_throw_slowdown_factor
    total_effective_frames = effective_pre_frames + effective_post_frames
    
    def update_with_slowdown(frame):
        if frame < effective_pre_frames:
            # Pre-throw phase
            pre_frame_idx = frame
            actual_pre_frame = min(pre_frame_idx // pre_throw_slowdown_factor, pre_throw_frames - 1)
            current_frame = actual_pre_frame + 1
            
            def_before_line.set_data(defender_before['y'][:current_frame], defender_before['x'][:current_frame])
            rec_before_line.set_data(receiver_before['y'][:current_frame], receiver_before['x'][:current_frame])
            
            # Update moving player markers with smooth interpolation
            alpha = (pre_frame_idx % pre_throw_slowdown_factor) / pre_throw_slowdown_factor
            if actual_pre_frame >= pre_throw_frames - 1:
                def_x = defender_before['x'].iloc[-1]
                def_y = defender_before['y'].iloc[-1]
                rec_x = receiver_before['x'].iloc[-1]
                rec_y = receiver_before['y'].iloc[-1]
            else:
                i = actual_pre_frame
                def_x = (1 - alpha) * defender_before['x'].iloc[i] + alpha * defender_before['x'].iloc[i + 1]
                def_y = (1 - alpha) * defender_before['y'].iloc[i] + alpha * defender_before['y'].iloc[i + 1]
                rec_x = (1 - alpha) * receiver_before['x'].iloc[i] + alpha * receiver_before['x'].iloc[i + 1]
                rec_y = (1 - alpha) * receiver_before['y'].iloc[i] + alpha * receiver_before['y'].iloc[i + 1]
            
            def_marker.set_offsets([[def_y, def_x]])
            rec_marker.set_offsets([[rec_y, rec_x]])
            
            def_after_line.set_data([], [])
            rec_after_line.set_data([], [])
            jump_line.set_data([], [])
            jump_line.set_visible(False)
            distance_line_plot.set_visible(False)
            
            artists = [def_before_line, rec_before_line, def_after_line, rec_after_line,
                       jump_line, distance_line_plot, expected_line,
                       def_marker, rec_marker]
            if jump_zone_area is not None:
                artists.append(jump_zone_area)
            return tuple(artists)
        else:
            # Post-throw phase
            post_frame_idx = frame - effective_pre_frames
            actual_post_frame = min(post_frame_idx // post_throw_slowdown_factor, post_throw_frames - 1)
            current_frame = actual_post_frame + 1
            
            # Show pre-throw lines completely
            def_before_line.set_data(defender_before['y'], defender_before['x'])
            rec_before_line.set_data(receiver_before['y'], receiver_before['x'])
            
            # Animate post-throw lines
            def_after_line.set_data(defender_after['y'][:current_frame], defender_after['x'][:current_frame])
            rec_after_line.set_data(receiver_after['y'][:current_frame], receiver_after['x'][:current_frame])
            
            # Update moving player markers with smooth interpolation
            alpha = (post_frame_idx % post_throw_slowdown_factor) / post_throw_slowdown_factor
            if actual_post_frame >= post_throw_frames - 1:
                def_x = defender_after['x'].iloc[-1]
                def_y = defender_after['y'].iloc[-1]
                rec_x = receiver_after['x'].iloc[-1]
                rec_y = receiver_after['y'].iloc[-1]
            elif actual_post_frame == 0 and post_frame_idx == 0:
                def_x = defender_after['x'].iloc[0]
                def_y = defender_after['y'].iloc[0]
                rec_x = receiver_after['x'].iloc[0]
                rec_y = receiver_after['y'].iloc[0]
            else:
                i = actual_post_frame
                def_x = (1 - alpha) * defender_after['x'].iloc[i] + alpha * defender_after['x'].iloc[i + 1]
                def_y = (1 - alpha) * defender_after['y'].iloc[i] + alpha * defender_after['y'].iloc[i + 1]
                rec_x = (1 - alpha) * receiver_after['x'].iloc[i] + alpha * receiver_after['x'].iloc[i + 1]
                rec_y = (1 - alpha) * receiver_after['y'].iloc[i] + alpha * receiver_after['y'].iloc[i + 1]

            def_marker.set_offsets([[def_y, def_x]])
            rec_marker.set_offsets([[rec_y, rec_x]])

            
            # Show post-throw lines
            if ball_land_x is not None and ball_land_y is not None:
                jump_line.set_data([receiver_after['y'].iloc[-1], ball_land_y],
                                               [receiver_after['x'].iloc[-1], ball_land_x])
                jump_line.set_visible(True)
            
            # Update distance graph
            distance_line_plot.set_data(defender_after['frame_id'][:current_frame], defender_after['distance_to_line'][:current_frame])
            distance_line_plot.set_markersize(2)
            distance_line_plot.set_visible(True)
            
            artists = [def_before_line, rec_before_line, def_after_line, rec_after_line,
                      jump_line, distance_line_plot, expected_line,
                      def_marker, rec_marker]
            if jump_zone_area is not None:
                artists.append(jump_zone_area)
            return tuple(artists)
    
    ani = animation.FuncAnimation(fig, update_with_slowdown,
                              frames=total_effective_frames,
                              blit=False,
                              interval=40)   # ~25 FPS

    ani.save(save_path, writer='pillow', dpi=150)
    print(f"Animation saved to {save_path}")


if __name__ == "__main__":
    # Specify the game_id and play_id to visualize
    GAME_ID = 2023100100
    PLAY_ID = 1429
    
    before_data = load_before_data()
    after_data = load_after_data()
    supplementary_data = load_supplementary_data()

    input_filtered_data = before_data[(before_data['game_id'] == GAME_ID) & (before_data['play_id'] == PLAY_ID)]
    output_filtered_data = after_data[(after_data['game_id'] == GAME_ID) & (after_data['play_id'] == PLAY_ID)]
    
    # Get supplementary row for title
    selected_row = None
    supp_match = supplementary_data[(supplementary_data['game_id'] == GAME_ID) & 
                                  (supplementary_data['play_id'] == PLAY_ID)]
    if not supp_match.empty:
        selected_row = supp_match.iloc[0]
    
    if input_filtered_data.empty or output_filtered_data.empty:
        print(f"No data found for game_id: {GAME_ID}, play_id: {PLAY_ID}")
        exit()

    # Get expected distance
    expected_dist = get_expected_distance(GAME_ID, PLAY_ID)
    if expected_dist is None:
        print(f"Warning: No expected distance found for game_id: {GAME_ID}, play_id: {PLAY_ID}")
        exit()
        
    print(f"Visualizing game_id: {GAME_ID}, play_id: {PLAY_ID}")
    animate_play(input_filtered_data, output_filtered_data, selected_row, expected_distance=expected_dist)